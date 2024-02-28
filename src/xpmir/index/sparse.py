"""Index for sparse models"""

import heapq
import torch
from queue import Empty
import torch.multiprocessing as mp
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Generic, Iterator, Union
from attrs import define
from experimaestro import (
    Annotated,
    Config,
    Task,
    Param,
    Meta,
    pathgenerator,
    tqdm,
    Constant,
)
from datamaestro_text.data.ir import DocumentRecord, DocumentStore, TextItem
from xpmir.learning import ModuleInitMode
from xpmir.learning.batchers import Batcher
from xpmir.utils.utils import batchiter, easylog
from xpmir.letor import Device, DeviceInformation, DEFAULT_DEVICE
from xpmir.text.encoders import TextEncoderBase, TextsRepresentationOutput, InputType
from xpmir.rankers import Retriever, TopicRecord, ScoredDocument
from xpmir.utils.iter import MultiprocessIterator
from xpmir.utils.multiprocessing import StoppableQueue
import xpmir_rust

logger = easylog()

# --- Index and retriever


class SparseRetrieverIndex(Config):
    index_path: Meta[Path]
    documents: Param[DocumentStore]

    index: xpmir_rust.index.SparseBuilderIndex
    ordered = False

    def initialize(self, in_memory: bool):
        self.index = xpmir_rust.index.SparseBuilderIndex.load(
            str(self.index_path.absolute()), in_memory
        )

    def retrieve(self, query: Dict[int, float], top_k: int) -> List[ScoredDocument]:
        results = []
        for sd in self.index.search_maxscore(query, top_k):
            results.append(
                ScoredDocument(
                    self.documents.document_int(sd.docid),
                    sd.score,
                )
            )

        return results


class SparseRetriever(Retriever, Generic[InputType]):
    index: Param[SparseRetrieverIndex]
    encoder: Param[TextEncoderBase[InputType, torch.Tensor]]
    topk: Param[int]

    batcher: Meta[Batcher] = Batcher()
    """The way to prepare batches of queries (when using retrieve_all)"""

    batchsize: Meta[int]
    """Size of batches (when using retrieve_all)"""

    in_memory: Meta[bool] = False
    """Whether the index should be fully loaded in memory (otherwise, uses
    virtual memory)"""

    def initialize(self):
        super().initialize()
        self.encoder.initialize(ModuleInitMode.DEFAULT.to_options(None))
        self.index.initialize(self.in_memory)

    def retrieve_all(
        self, queries: Dict[str, InputType]
    ) -> Dict[str, List[ScoredDocument]]:
        """Input queries: {id: text}"""

        def reducer(
            batch: List[Tuple[str, InputType]],
            results: Dict[str, List[ScoredDocument]],
            progress,
        ):
            for (key, _), vector in zip(
                batch,
                self.encoder([text for _, text in batch]).value.cpu().detach().numpy(),
            ):
                (ix,) = vector.nonzero()
                query = {ix: float(v) for ix, v in zip(ix, vector[ix])}
                results[key] = self.index.retrieve(query, self.topk)
                progress.update(1)
            return results

        self.encoder.eval()
        batcher = self.batcher.initialize(self.batchsize)
        results = {}
        items = list(queries.items())
        with tqdm(
            desc="Retrieve documents", total=len(items), unit="queries"
        ) as progress:
            with torch.no_grad():
                for batch in batchiter(self.batchsize, items):
                    results = batcher.reduce(batch, reducer, results, progress)

        return results

    def retrieve(self, query: TopicRecord, top_k=None) -> List[ScoredDocument]:
        """Search with document-at-a-time (DAAT) strategy

        :param top_k: Overrides the default top-K value
        """

        # Build up iterators
        vector = self.encoder([query]).value[0].cpu().detach().numpy()
        (ix,) = vector.nonzero()  # ix represents the position without 0 in the vector
        query = {
            ix: float(v) for ix, v in zip(ix, vector[ix])
        }  # generate a dict: {position:value}
        return self.index.retrieve(query, top_k or self.topk)


@define(frozen=True)
class EncodedDocument:
    docid: int
    value: torch.Tensor


@define(frozen=True)
class DocumentRange:
    rank: int
    start: int
    end: int

    def __lt__(self, other: "DocumentRange"):
        return self.start < other.start


class SparseRetrieverIndexBuilder(Task, Generic[InputType]):
    """Builds an index from a sparse representation

    Assumes that document and queries have the same dimension, and
    that the score is computed through an inner product
    """

    documents: Param[DocumentStore]
    """Set of documents to index"""

    encoder: Param[TextEncoderBase[InputType, TextsRepresentationOutput]]
    """The encoder"""

    batcher: Meta[Batcher] = Batcher()
    """Batcher used when computing representations"""

    batch_size: Param[int]
    """Size of batches"""

    ordered_index: Param[bool]
    """Ordered index: if not ordered, use DAAT strategy (WAND), otherwise, use
    fast top-k strategies"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device for building the index"""

    max_postings: Meta[int] = 16384
    """Maximum number of postings (per term) before flushing to disk"""

    index_path: Annotated[Path, pathgenerator("index")]

    in_memory: Meta[bool] = False
    """Whether the index should be fully loaded in memory (otherwise, uses
    virtual memory)"""

    version: Constant[int] = 3
    """Version 3 of the index"""

    max_docs: Param[int] = 0
    """Maximum number of indexed documents"""

    def task_outputs(self, dep):
        """Returns a sparse retriever index that can be used by a
        SparseRetriever to search efficiently for documents"""

        return dep(
            SparseRetrieverIndex(index_path=self.index_path, documents=self.documents)
        )

    def execute(self):
        max_docs = (
            self.documents.documentcount
            if self.max_docs == 0
            else min(self.max_docs, self.documents.documentcount)
        )

        iter_batches = tqdm(
            MultiprocessIterator(
                batchiter(
                    self.batch_size,
                    zip(
                        range(sys.maxsize if self.max_docs == 0 else self.max_docs),
                        MultiprocessIterator(self.documents.iter_documents()).start(),
                    ),
                )
            ),
            total=max_docs // self.batch_size,
            unit_scale=self.batch_size,
            unit="documents",
            desc="Building the index",
        )

        self.encoder.initialize(ModuleInitMode.DEFAULT.to_options(None))

        closed = mp.Event()
        queues = [
            StoppableQueue(2 * self.batch_size + 1, closed)
            for _ in range(self.device.n_processes)
        ]

        # Cleanup the index before starting
        from shutil import rmtree

        if self.index_path.is_dir():
            rmtree(self.index_path)
        self.index_path.mkdir(parents=True)

        # Start the index process
        index_process = mp.Process(
            target=self.index,
            args=(queues,),
            daemon=True,
        )
        index_process.start()

        # Waiting for the encoder process to end
        logger.info(f"Starting to index {max_docs} documents")

        try:
            self.device.execute(self.device_execute, iter_batches, queues)
        finally:
            logger.info("Waiting for the index process to stop")
            index_process.join()
            if index_process.exitcode != 0:
                logger.warning(
                    "Indexer process has finished with exit code %d",
                    index_process.exitcode,
                )
                raise RuntimeError("Failure")

    def index(
        self, queues: List[StoppableQueue[Union[DocumentRange, EncodedDocument]]]
    ):
        """Index encoded documents

        :param queues: Queues are used to send tensors
        """
        try:
            # Get ranges
            logger.info(
                "Starting the indexing process (%d queues) in %s",
                len(queues),
                self.index_path,
            )
            indexer = xpmir_rust.index.SparseIndexer(str(self.index_path))
            heap = [queue.get() for queue in queues]
            heapq.heapify(queues)

            # Loop over them
            while heap:
                # Process current range
                current = heap[0]
                logger.debug("Handling range: %s", current)
                for docid in range(current.start, current.end + 1):
                    encoded = queues[current.rank].get()
                    assert (
                        encoded.docid == docid
                    ), f"Mismatch in document IDs ({encoded.docid} vs {docid})"

                    (nonzero_ix,) = encoded.value.nonzero()
                    indexer.add(
                        docid, nonzero_ix.astype(np.uint64), encoded.value[nonzero_ix]
                    )

                # Get next range
                next_range = queues[current.rank].get()  # type: DocumentRange
                if next_range:
                    heapq.heappushpop(heap, next_range)
                else:
                    logger.info("Iterator %d is over", current.rank)
                    heapq.heappop(heap)

            logger.info("Building the index")
            indexer.build(self.in_memory)
        except Empty:
            logger.warning("One encoder got a problem... stopping")
            raise
        except Exception:
            # Close all the queues
            logger.exception(
                "Got an exception in the indexing process, closing the queues"
            )
            queues[0].stop()
            raise

    def device_execute(
        self,
        device_information: DeviceInformation,
        iter_batches: Iterator[List[Tuple[int, DocumentRecord]]],
        queues: List[StoppableQueue],
    ):
        try:
            # Encode all documents
            logger.info(
                "Load the encoder and "
                f"transfer to the target device {self.device.value}"
            )

            encoder = self.encoder.to(self.device.value).eval()
            queue = queues[device_information.rank]
            batcher = self.batcher.initialize(self.batch_size)

            # Index
            with torch.no_grad():
                for batch in iter_batches:
                    # Signals the output range
                    queue.put(
                        DocumentRange(
                            device_information.rank, batch[0][0], batch[-1][0]
                        )
                    )
                    # Outputs the documents
                    batcher.process(batch, self.encode_documents, encoder, queue)

            # Build the index
            logger.info("Closing queue %d", device_information.rank)
            queue.put(None)
        except Exception:
            queue.stop()
            raise

    def encode_documents(
        self,
        batch: List[Tuple[int, DocumentRecord]],
        encoder: TextEncoderBase[InputType, TextsRepresentationOutput],
        queue: "mp.Queue[EncodedDocument]",
    ):
        # Assumes for now dense vectors
        vectors = (
            encoder([d[TextItem].text for _, d in batch]).value.cpu().numpy()
        )  # bs * vocab
        for vector, (docid, _) in zip(vectors, batch):
            queue.put(EncodedDocument(docid, vector))
