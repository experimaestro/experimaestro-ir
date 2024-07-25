"""Index for sparse models"""

import asyncio
from functools import cached_property
import threading
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
from datamaestro_text.data.ir import DocumentRecord, DocumentStore
from xpmir.learning import ModuleInitMode
from xpmir.learning.batchers import Batcher
from xpmir.utils.utils import batchiter, easylog
from xpmir.letor import Device, DeviceInformation, DEFAULT_DEVICE
from xpmir.text.encoders import TextEncoderBase, TextsRepresentationOutput, InputType
from xpmir.rankers import Retriever, TopicRecord, ScoredDocument
from xpmir.utils.iter import MultiprocessIterator
from xpmir.utils.multiprocessing import StoppableQueue, available_cpus
import impact_index

logger = easylog()

# --- Index and retriever


class SparseRetrieverIndex(Config):
    index_path: Meta[Path]
    documents: Param[DocumentStore]

    index: impact_index.Index
    ordered = False

    def initialize(self, in_memory: bool):
        self.index = impact_index.Index.load(str(self.index_path.absolute()), in_memory)

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

    async def aio_retrieve(
        self, query: Dict[int, float], top_k: int
    ) -> List[ScoredDocument]:
        results = []
        for sd in await self.index.aio_search_maxscore(query, top_k):
            results.append(
                ScoredDocument(
                    self.documents.document_int(sd.docid),
                    sd.score,
                )
            )

        return results


class SparseRetriever(Retriever, Generic[InputType]):
    index: Param[SparseRetrieverIndex]
    """The sparse retriever index"""

    encoder: Param[TextEncoderBase[InputType, TextsRepresentationOutput]]
    """Encodes InputType records to text representation output"""

    topk: Param[int]
    """Number of documents to return"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device for building the index"""

    batcher: Meta[Batcher] = Batcher()
    """The way to prepare batches of queries (when using retrieve_all)"""

    batchsize: Meta[int]
    """Size of batches (when using retrieve_all)"""

    in_memory: Meta[bool] = False
    """Whether the index should be fully loaded in memory (otherwise, uses
    virtual memory)"""

    def initialize(self):
        super().initialize()
        logger.info("Initializing the encoder")
        self.encoder.initialize(ModuleInitMode.DEFAULT.to_options(None))
        self.encoder.to(self.device.value)
        logger.info("Initializing the index")
        self.index.initialize(self.in_memory)

    def retrieve_all(
        self, queries: Dict[str, InputType]
    ) -> Dict[str, List[ScoredDocument]]:
        """Input queries: {id: text}"""

        async def aio_search_worker(progress, results: Dict, queue: asyncio.Queue):
            try:
                while True:
                    key, query, topk = await queue.get()
                    results[key] = await self.index.aio_retrieve(query, topk)
                    progress.update(1)
                    queue.task_done()
            except asyncio.exceptions.CancelledError:
                # Just stopped
                pass
            except Exception:
                logger.exception("Error in worker thread")

        async def reducer(
            batch: List[Tuple[str, InputType]],
            queue: asyncio.Queue,
        ):
            x = self.encoder([topic for _, topic in batch]).value.cpu().detach().numpy()
            assert len(x) == len(batch), (
                f"Discrepancy between counts of vectors ({len(x)})"
                f" and number queries ({len(batch)})"
            )
            for (key, _), vector in zip(batch, x):
                (ix,) = vector.nonzero()
                query = {ix: float(v) for ix, v in zip(ix, vector[ix])}
                logger.debug("Adding topic %s to the queue", key)
                await queue.put((key, query, self.topk))
                logger.debug("[done] Adding topic %s to the queue", key)

        async def aio_process():
            workers = []
            results = {}
            try:
                queue = asyncio.Queue(available_cpus())
                items = list(queries.items())

                with tqdm(
                    desc="Retrieve documents", total=len(items), unit="queries"
                ) as progress:
                    self.encoder.eval()
                    for _ in range(available_cpus()):
                        worker = asyncio.create_task(
                            aio_search_worker(progress, results, queue)
                        )
                        workers.append(worker)

                    batcher = self.batcher.initialize(self.batchsize)
                    with torch.no_grad():
                        for batch in batchiter(self.batchsize, items):
                            await batcher.aio_reduce(batch, reducer, queue)

                    # Just wait for this to end
                    await queue.join()

            finally:
                # Stop all retriever workers
                for worker in workers:
                    worker.cancel()
            return results

        logger.info("Retrieve all with %d CPUs", available_cpus())
        results = asyncio.run(aio_process())
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


class DocumentIterator:
    def __init__(self, documents, max_docs, batch_size):
        self.documents = documents
        self.max_docs = max_docs
        self.batch_size = batch_size

    @cached_property
    def iterator(self):
        return batchiter(
            self.batch_size,
            zip(
                range(self.max_docs or sys.maxsize),
                self.documents.iter_documents(),
            ),
        )

    def __next__(self):
        return next(self.iterator)


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
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")

        max_docs = 0
        if self.max_docs:
            max_docs = min(self.max_docs, self.documents.documentcount or sys.maxsize)
            logger.warning("Limited indexing to %d documents", max_docs)

        iter_batches = MultiprocessIterator(
            DocumentIterator(self.documents, max_docs, self.batch_size)
        ).detach()

        self.encoder.initialize(ModuleInitMode.DEFAULT.to_options(None))

        closed = mp.Event()
        queues = [
            StoppableQueue(2 * self.batch_size + 1, closed)
            for _ in range(self.device.n_processes)
        ]

        # Cleanup the index before starting
        # ENHANCE: recover index build when possible
        from shutil import rmtree

        if self.index_path.is_dir():
            rmtree(self.index_path)
        self.index_path.mkdir(parents=True)

        # Start the index process (thread)
        index_thread = threading.Thread(
            target=self.index,
            name="index",
            args=(queues, max_docs),
        )
        index_thread.start()

        # Waiting for the encoder process to end
        logger.info(f"Starting to index {max_docs} documents")

        try:
            self.device.execute(
                self.device_execute,
                iter_batches,
                self.encoder,
                self.batcher,
                self.batch_size,
                queues,
            )
        except Exception:
            logger.exception("Got an exception while running encoders")
        finally:
            logger.info("Waiting for the index process to stop")
            index_thread.join()
            if not self.index_done:
                raise RuntimeError("Indexing thread did not complete")

    def index(
        self,
        queues: List[StoppableQueue[Union[DocumentRange, EncodedDocument]]],
        max_docs: int,
    ):
        """Index encoded documents

        :param queues: Queues are used to send tensors
        """
        self.index_done = False
        with tqdm(
            total=max_docs,
            unit="documents",
            desc="Building the index",
        ) as pb:
            try:
                # Get ranges
                logger.info(
                    "Starting the indexing process (%d queues) in %s",
                    len(queues),
                    self.index_path,
                )
                indexer = impact_index.IndexBuilder(str(self.index_path))
                heap = [queue.get() for queue in queues]
                heapq.heapify(heap)

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
                            docid,
                            nonzero_ix.astype(np.uint64),
                            encoded.value[nonzero_ix],
                        )
                        pb.update()

                    # Get next range
                    next_range = queues[current.rank].get()  # type: DocumentRange
                    if next_range:
                        logger.debug("Got next range: %s", next_range)
                        heapq.heappushpop(heap, next_range)
                    else:
                        logger.info("Iterator %d is over", current.rank)
                        heapq.heappop(heap)

                logger.info("Building the index")
                indexer.build(self.in_memory)

                logger.info("Index built")
                self.index_done = True
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

    @staticmethod
    def device_execute(
        device_information: DeviceInformation,
        iter_batches: Iterator[List[Tuple[int, DocumentRecord]]],
        encoder,
        batcher,
        batch_size,
        queues: List[StoppableQueue],
    ):
        try:
            # Encode all documents
            logger.info(
                "Load the encoder and "
                f"transfer to the target device {device_information.device}"
            )

            encoder = encoder.to(device_information.device).eval()
            queue = queues[device_information.rank]
            batcher = batcher.initialize(batch_size)

            # Index
            with torch.no_grad():
                for batch in iter_batches:
                    # Signals the output range
                    document_range = DocumentRange(
                        device_information.rank, batch[0][0], batch[-1][0]
                    )
                    logger.debug(
                        "Starting range [%d] %s",
                        device_information.rank,
                        document_range,
                    )
                    queue.put(document_range)

                    # Outputs the documents
                    batcher.process(
                        batch,
                        SparseRetrieverIndexBuilder.encode_documents,
                        encoder,
                        queue,
                    )

            # Build the index
            logger.info("Closing queue %d", device_information.rank)
            queue.put(None)
        except Exception:
            queue.stop()
            raise

    @staticmethod
    def encode_documents(
        batch: List[Tuple[int, DocumentRecord]],
        encoder: TextEncoderBase[InputType, TextsRepresentationOutput],
        queue: "mp.Queue[EncodedDocument]",
    ):
        # Assumes for now dense vectors
        vectors = encoder([d for _, d in batch]).value.cpu().numpy()  # bs * vocab
        for vector, (docid, _) in zip(vectors, batch):
            queue.put(EncodedDocument(docid, vector))
