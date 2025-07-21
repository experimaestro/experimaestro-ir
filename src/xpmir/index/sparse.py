"""Index for sparse models"""

from abc import abstractmethod, ABC
import asyncio
from functools import cached_property
import logging
import shutil
import threading
import heapq
import torch
from queue import Empty, Queue
import torch.multiprocessing as mp
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Generic, Iterator, Union, Any
from attrs import define
from experimaestro import (
    Config,
    Task,
    Param,
    Meta,
    field,
    PathGenerator,
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
    def __init__(
        self, documents: DocumentStore, last_doc_id: None | int, max_docs, batch_size
    ):
        self.documents = documents
        self.last_doc_id = last_doc_id
        self.max_docs = max_docs
        self.batch_size = batch_size

    @cached_property
    def iterator(self):
        start = 0 if self.last_doc_id is None else self.last_doc_id + 1
        iter = self.documents.iter_documents_from(start)

        return batchiter(
            self.batch_size,
            zip(
                range(start, self.max_docs or sys.maxsize),
                iter,
            ),
        )

    def __next__(self):
        return next(self.iterator)


class AbstractSparseRetrieverIndexBuilder(Task, ABC, Generic[InputType]):
    """Builds an index from a sparse representation

    Assumes that document and queries have the same dimension, and
    that the score is computed through an inner product
    """

    documents: Param[DocumentStore]
    """Set of documents to index"""

    encoder: Param[TextEncoderBase[InputType, TextsRepresentationOutput]]
    """The encoder"""

    batcher: Meta[Batcher] = field(default_factory=Batcher.C)
    """Batcher used when computing representations"""

    batch_size: Param[int]
    """Size of batches"""

    ordered_index: Param[bool]
    """Ordered index: if not ordered, use DAAT strategy (WAND), otherwise, use
    fast top-k strategies"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device for building the index"""

    version: Constant[int] = 3
    """Version 3 of the index"""

    max_docs: Param[int] = 0
    """Maximum number of indexed documents"""

    def execute(self):
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")

        max_docs = 0
        if self.max_docs:
            max_docs = min(self.max_docs, self.documents.documentcount or sys.maxsize)
            logger.warning("Limited indexing to %d documents", max_docs)

        self.encoder.initialize(ModuleInitMode.DEFAULT.to_options(None))

        closed = mp.Event()
        queues = [
            StoppableQueue(2 * self.batch_size + 1, closed)
            for _ in range(self.device.n_processes)
        ]

        # Start the index process (thread)
        last_doc_id_queue = Queue()
        index_thread = threading.Thread(
            target=self.index,
            name="index",
            args=(queues, max_docs, last_doc_id_queue),
        )
        index_thread.start()

        last_doc_id = last_doc_id_queue.get()
        logger.info(f"Starting to index {max_docs} documents")
        if last_doc_id:
            logging.info(f" -- recovering from last indexed document: {last_doc_id}")

        iter_batches = MultiprocessIterator(
            DocumentIterator(self.documents, last_doc_id, max_docs, self.batch_size)
        ).detach()

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
        last_doc_id_queue: Queue[Optional[int]],
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

                if last_doc_id := self.create_index_builder():
                    logger.info("Starting back from document %d", last_doc_id)
                    pb.update(last_doc_id)

                logging.info("Last indexed document: %s", last_doc_id)
                last_doc_id_queue.put(last_doc_id)

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
                        self.add_encoded_document(docid, encoded, nonzero_ix)
                        pb.update()

                    # Get next range
                    # type: DocumentRange
                    next_range = queues[current.rank].get()
                    if next_range:
                        logger.debug("Got next range: %s", next_range)
                        heapq.heappushpop(heap, next_range)
                    else:
                        logger.info("Iterator %d is over", current.rank)
                        heapq.heappop(heap)

                logger.info("Building the index")
                self.build_index()

                logger.info("Index built")
                self.index_done = True
            except Empty:
                logger.warning("One encoder got a problem... stopping")
                raise
            except BaseException:
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
        except BaseException:
            logging.exception("Got an exception while encoding documents")
            queue.stop()
            raise

    @staticmethod
    def encode_documents(
        batch: List[Tuple[int, DocumentRecord]],
        encoder: TextEncoderBase[InputType, TextsRepresentationOutput],
        queue: "Queue[EncodedDocument]",
    ):
        # Assumes for now dense vectors
        vectors = encoder([d for _, d in batch]).value.cpu().numpy()  # bs * vocab
        for vector, (docid, _) in zip(vectors, batch):
            queue.put(EncodedDocument(docid, vector))

    @abstractmethod
    def build_index(self):
        ...

    @abstractmethod
    def add_encoded_document(self, docid, encoded, nonzero_ix):
        ...

    @abstractmethod
    def create_index_builder(self) -> int | None:
        """Creates a new index builder

        :return: the last doc ID when using checkpointing (so we start back with
            the next document)
        """
        ...


class SparseRetriever(Retriever, Generic[InputType]):
    index: Param["AbstractSparseRetrieverIndex"]
    """The sparse retriever index"""

    encoder: Param[TextEncoderBase[InputType, TextsRepresentationOutput]]
    """Encodes InputType records to text representation output"""

    topk: Param[int]
    """Number of documents to return"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device for building the index"""

    batcher: Meta[Batcher] = field(default_factory=Batcher.C)
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
                    results[key] = await self.index.aio_retrieve(
                        query, topk, **self.get_retrieval_options()
                    )
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
        return self.index.retrieve(
            query, top_k or self.topk, **self.get_retrieval_options()
        )

    def get_retrieval_options(self) -> dict[str, Any]:
        """Returns extra retrieval options to be used when retrieving"""
        return {}

    def __validate__(self):
        # Checks that we are using the right retriever
        assert isinstance(
            self, self.index.Retriever
        ), f"{type(self)} is not an instance of {self.index.Retriever}"


class AbstractSparseRetrieverIndex(Config, ABC):
    Retriever = SparseRetriever

    documents: Param[DocumentStore]
    """The indexed document collection"""

    index: impact_index.Index
    ordered = False

    def initialize(self, in_memory: bool):
        self.index = impact_index.Index.load(str(self.index_path.absolute()), in_memory)

    @abstractmethod
    def retrieve(
        self, query: Dict[int, float], top_k: int, **kwargs
    ) -> List[ScoredDocument]:
        ...

    @abstractmethod
    async def aio_retrieve(
        self, query: Dict[int, float], top_k: int, **kwargs
    ) -> List[ScoredDocument]:
        ...


# ---
# --- CIFF file
# ---


class CIFFBuilder:
    def __init__(self, path: Path, levels: int):
        """Constructs a CIFF index

        :param path: _description_
        :param levels: _description_
        """
        # Path of the final index ("file.ciff")
        self.path = path


# ---
# --- Sparse index with the impact_index library
# ---


class SparseRetrieverIndex(AbstractSparseRetrieverIndex):
    index_path: Meta[Path]

    ordered = False

    index: impact_index.Index

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


class SparseRetrieverIndexBuilder(AbstractSparseRetrieverIndexBuilder[InputType]):
    in_memory: Meta[bool] = False
    """Whether the index should be fully loaded in memory (otherwise, uses
    virtual memory)"""

    index_path: Meta[Path] = field(default_factory=PathGenerator("index"))

    checkpoint_frequency: Meta[int] = 0
    """Checkpoint frequency (allows recovery at the cost of writing some
    information to disk)"""

    max_postings: Meta[Optional[int]] = None
    """Number of postings before dumping a term postings to disk"""

    def task_outputs(self, dep):
        """Returns a sparse retriever index that can be used by a
        SparseRetriever to search efficiently for documents"""

        return dep(
            SparseRetrieverIndex.C(index_path=self.index_path, documents=self.documents)
        )

    def create_index_builder(self):
        if self.index_path.is_dir() and self.checkpoint_frequency == 0:
            # Removing partially built index
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Setup options
        options = impact_index.BuilderOptions()
        options.checkpoint_frequency = self.checkpoint_frequency
        if self.max_postings is not None:
            options.in_memory_threshold = self.max_postings

        # and create the index builder
        self.indexer = impact_index.IndexBuilder(str(self.index_path), options)
        return self.indexer.get_checkpoint_doc_id()

    def build_index(self):
        self.indexer.build(self.in_memory)

    def add_encoded_document(self, docid, encoded, nonzero_ix):
        self.indexer.add(
            docid,
            nonzero_ix.astype(np.uint64),
            encoded.value[nonzero_ix],
        )


# ---
# --- Sparse index with the BMP library
# --- https://github.com/pisa-engine/BMP
# ---


class BMPSparseRetriever(SparseRetriever):
    """A Block-Max Pruning retriever"""

    alpha: Param[float]
    """Granularity of approximation (0 to 1, 1 = no approximation)"""

    beta: Param[float]
    """Percentage of query tokens to keep (0 to 1, 1 = no pruning)"""

    def get_retrieval_options(self) -> dict[str, Any]:
        """Returns extra retrieval options to be used when retrieving"""
        return {"alpha": self.alpha, "beta": self.beta}



class BMPSparseRetrieverIndex(AbstractSparseRetrieverIndex):
    Retriever = BMPSparseRetriever

    index_path: Meta[Path]
    """The path of the BMP index"""

    index: impact_index.Index

    def initialize(self, in_memory: bool):
        if not in_memory:
            logger.warning("BMP indices are in-memory only")
        try:
            from bmp import Searcher
        except ModuleNotFoundError:
            logger.warning(
                "Did not find the Block Max Pruning (bmp) library. "
                "Check https://github.com/pisa-engine/BMP"
            )
            raise
        logger.info("Loading BMP index from %s", self.index_path)
        self.searcher = Searcher(str(self.index_path))

    def retrieve(
        self, query: Dict[int, float], top_k: int, alpha=None, beta=None
    ) -> List[ScoredDocument]:
        results = []
        doc_ids, scores = self.searcher.search(
            {str(ix): float(value) for ix, value in query.items()},
            k=top_k,
            alpha=alpha,
            beta=beta,
        )

        for doc_id, score in zip(doc_ids, scores):
            results.append(
                ScoredDocument(
                    self.documents.document_int(int(doc_id)),
                    score,
                )
            )

        return results

    async def aio_retrieve(
        self, query: Dict[int, float], top_k: int, **kwargs
    ) -> List[ScoredDocument]:
        return self.retrieve(query, top_k, **kwargs)

class Sparse2BMPConverter(Task):
    index: Param[SparseRetrieverIndex]
    """The sparse index"""

    bmp_index_path: Meta[Path] = field(default_factory=PathGenerator("index.bmp"))
    """The final index path"""

    block_size: Param[int]
    """The block size"""

    compress_range: Param[bool]
    """Compress the BM index"""

    def task_outputs(self, dep):
        """Returns a sparse retriever index that can be used by a
        SparseRetriever to search efficiently for documents"""

        return dep(
            BMPSparseRetrieverIndex.C(
                index_path=self.bmp_index_path, documents=self.index.documents
            )
        )

    def execute(self):
        logging.info("Loading the index")
        self.index.initialize(False)

        logging.info("Converting to BMP")
        self.index.index.to_bmp(str(self.bmp_index_path), self.block_size, self.compress_range)

        logging.info("Done")


class BMPSparseRetrieverIndexBuilder(SparseRetrieverIndexBuilder[InputType]):
    """Index using a BMP index
    """
    block_size: Param[int]
    """The block size"""

    compress_range: Param[bool]
    """Compress the BM index"""

    bmp_index_path: Meta[Path] = field(default_factory=PathGenerator("index.bmp"))
    """The final index path"""

    def task_outputs(self, dep):
        """Returns a sparse retriever index that can be used by a
        SparseRetriever to search efficiently for documents"""

        return dep(
            BMPSparseRetrieverIndex.C(
                index_path=self.bmp_index_path, documents=self.documents
            )
        )

    def build_index(self):
        # Build the index
        logger.info("Building the index")
        index = self.indexer.build(False)

        logger.info("Converting to BMP index")
        index.to_bmp(str(self.bmp_index_path), self.block_size, self.compress_range)

        # Removes the old index
        logger.info("Removing the old index path")
        index = None
        shutil.rmtree(self.index_path, ignore_errors=True)
