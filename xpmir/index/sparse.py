"""Index for sparse models"""

from dataclasses import dataclass
import torch
import numpy as np
from pathlib import Path
from typing import Callable, Iterator, List, BinaryIO, NamedTuple, Tuple
from experimaestro import Annotated, Config, Task, Param, Meta, pathgenerator, tqdm
from datamaestro_text.data.ir import AdhocDocument, AdhocDocumentStore
from xpmir.letor.batchers import Batcher
from xpmir.utils import batchiter, easylog
from xpmir.letor import Device, DEFAULT_DEVICE
from xpmir.text.encoders import TextEncoder
from xpmir.rankers import Retriever, ScoredDocument
import array
import pickle
import mmap
import heapq
from numba import typed

import numba
from numba.experimental import jitclass


logger = easylog()


class InternalScoredDocument:
    docid: int
    score: float

    def __init__(self, docid, score):
        self.docid = docid
        self.score = score

    def __lt__(self, other: "InternalScoredDocument"):
        return self.score < other.score


class Posting(NamedTuple):
    index: int
    """The iterator index"""

    docid: int
    """The document ID"""

    value: float
    """The value"""

    def __lt__(self, other: "Posting"):
        return self.docid < other.docid


def docidArray():
    return array.array("I")


def valuesArray():
    return array.array("f")


DOCID_TYPE = np.int64
VALUE_TYPE = np.float32

DOCID_ITEMSIZE = np.dtype(DOCID_TYPE).itemsize
VALUE_ITEMSIZE = np.dtype(VALUE_TYPE).itemsize


class SparseRetrieverIndex(Config):
    index_path: Meta[Path]
    info_path: Meta[Path]
    documents: Param[AdhocDocumentStore]

    starts: List[array.array]
    lengths: List[array.array]

    def __postinit__(self):
        with self.info_path.open("rb") as fp:
            self.starts, self.lengths = pickle.load(fp)

        self.mm_file = self.index_path.open("r+b")

        self.index_mm = mmap.mmap(self.mm_file.fileno(), 0)

    def iter_postings(self, index: int, term_ix: int) -> Iterator[Posting]:

        for start, length in zip(self.starts[term_ix], self.lengths[term_ix]):
            # Read from mmap
            self.index_mm.seek(start)
            docids = np.frombuffer(
                self.index_mm.read(DOCID_ITEMSIZE * length), dtype=DOCID_TYPE
            )  #  .read(docids.itemsize * length))
            values = np.frombuffer(
                self.index_mm.read(VALUE_ITEMSIZE * length), dtype=VALUE_TYPE
            )  #  .read(values.itemsize * length))
            # Output postings
            for docid, value in zip(docids, values):
                yield Posting(index, docid, value)


class TopDocuments:
    def __init__(self, topk: int):
        self.topk = topk
        self.dheap: List[InternalScoredDocument] = []

    def add(self, current: InternalScoredDocument):
        if current.docid == -1:
            return

        if len(self.dheap) < self.topk:
            heapq.heappush(self.dheap, current)
        elif current.score > self.dheap[0].score:
            heapq.heapreplace(self.dheap, current)


class SparseRetriever(Retriever):
    index: Param[SparseRetrieverIndex]
    encoder: Param[TextEncoder]
    topk: Param[int]

    def retrieve(self, query: str) -> List[ScoredDocument]:
        """Search with document-at-a-time (DAAT) strategy"""

        # Build up iterators
        vector = self.encoder([query])[0].cpu()
        query_values = []
        iterators: List[Iterator[Posting]] = []

        iterator_index = 0
        for ix in torch.nonzero(vector):
            iterators.append(self.index.iter_postings(iterator_index, int(ix)))
            query_values.append(vector[ix])
            iterator_index += 1

        # Build a heap for iterators / postings
        pheap = [next(iter) for iter in iterators]
        heapq.heapify(pheap)

        # Document heap
        top = TopDocuments(self.topk)

        # While we have posting to process
        current = InternalScoredDocument(-1, 0)
        while pheap:
            tip = pheap[0]
            v = tip.value * query_values[tip.index]
            if tip.docid == current.docid:
                current.score += v
            else:
                top.add(current)
                current = InternalScoredDocument(tip.docid, v)

            # Fetch next
            try:
                new_one = next(iterators[tip.index])
                heapq.heapreplace(pheap, new_one)
            except StopIteration:
                # Remove this iteator
                heapq.heappop(pheap)

        # Add last document
        top.add(current)

        top.dheap.sort(reverse=True)
        return [
            ScoredDocument(
                self.index.documents.docid_internal2external(d.docid), d.score, None
            )
            for d in top.dheap
        ]


@jitclass(
    [
        ("max_postings", numba.int64),
        ("values", numba.float32[:, :]),
        ("docids", numba.int64[:, :]),
        ("npostings", numba.int64[:]),
    ]
)
class Postings:
    def __init__(self, nterms, max_postings):
        self.max_postings = max_postings
        self.values = np.zeros(
            (nterms, max_postings), VALUE_TYPE
        )  # term x max. postings
        self.docids = np.zeros(
            (nterms, max_postings), DOCID_TYPE
        )  # term x max. postings
        self.npostings = np.zeros(
            nterms, np.int64
        )  # How many posting are stored (per term)


class SparseRetrieverIndexBuilder(Task):
    """Builds an index from a sparse representation

    Assumes that document and queries have the same dimension, and
    that the score is computed through an inner product
    """

    documents: Param[AdhocDocumentStore]
    """Set of documents to index"""

    encoder: Param[TextEncoder]
    """The encoder"""

    batcher: Meta[Batcher] = Batcher()
    """Batcher used when computing representations"""

    batch_size: Param[int]
    """Size of batches"""

    device: Meta[Device] = DEFAULT_DEVICE

    max_postings: Meta[int] = 16384
    """Maximum number of postings (per term) before flushing to disk"""

    index_path: Annotated[Path, pathgenerator("index.bin")]
    info_path: Annotated[Path, pathgenerator("info.bin")]

    def config(self):
        """Returns a sparse retriever index that can be used by a SparseRetriever to search efficiently
        for documents"""
        return SparseRetrieverIndex(
            index_path=self.index_path,
            info_path=self.info_path,
            documents=self.documents,
        )

    def execute(self):
        # Encode all documents
        logger.info(
            f"Loading the encoder and transfering to the target device {self.device.value}"
        )
        self.encoder.initialize()
        self.encoder.to(self.device.value).eval()

        batcher = self.batcher.initialize(self.batch_size)

        doc_iter = tqdm(
            self.documents.iter_documents(),
            total=self.documents.documentcount,
            desc="Building the index",
        )

        # Prepare the terms informations
        self.postings = Postings(self.encoder.dimension, self.max_postings)
        self.posting_starts = [array.array("I") for _ in range(self.encoder.dimension)]
        self.posting_lengths = [array.array("I") for _ in range(self.encoder.dimension)]

        logger.info(f"Starting to index {self.documents.documentcount} documents")
        with self.index_path.open("wb") as index_out:
            with torch.no_grad():
                for batch in batchiter(self.batch_size, doc_iter):
                    batcher.process(batch, self.encode_documents, index_out)

            # Build the final index
            for term_ix in np.nonzero(self.postings.npostings)[0]:
                self.flush(term_ix, index_out)

        logger.info("Dumping the index to %s", self.info_path)
        with self.info_path.open("wb") as fp:
            pickle.dump(
                (self.posting_starts, self.posting_lengths),
                fp,
            )

    def encode_documents(self, batch: List[AdhocDocument], index_out: BinaryIO):
        # Assumes for now dense vectors
        vectors = self.encoder([d.text for d in batch]).cpu().numpy()
        assert all(
            d.internal_docid is not None for d in batch
        ), f"No internal document ID provided by document store {type(self.documents)}"
        buffer = encode_documents_numba(
            vectors,
            np.array([d.internal_docid for d in batch], dtype=np.int64),
            self.postings,
        )

        for term_ix, docids, values in zip(
            buffer.term_ix, buffer.docids, buffer.values
        ):
            self.posting_starts[term_ix].append(index_out.tell())
            self.posting_lengths[term_ix].append(self.postings.max_postings)
            docids.tofile(index_out)
            values.tofile(index_out)

    def flush(self, term_ix: int, index_out: BinaryIO):
        """Flush term postings"""
        npostings = self.postings.npostings[term_ix]
        if npostings == 0:
            return

        self.posting_starts[term_ix].append(index_out.tell())
        self.posting_lengths[term_ix].append(npostings)

        self.postings.docids[term_ix, :npostings].tofile(index_out)
        self.postings.values[term_ix, :npostings].tofile(index_out)

        # Resets
        self.postings.npostings[term_ix] = 0


float_ndarray = numba.float32[:]
int64_ndarray = numba.int64[:]


@jitclass(
    [
        ("term_ix", numba.types.ListType(numba.uint64)),
        ("values", numba.types.ListType(float_ndarray)),
        ("docids", numba.types.ListType(int64_ndarray)),
    ]
)
class Buffer:
    def __init__(self):
        self.term_ix = typed.List.empty_list(numba.uint64)
        self.docids = typed.List.empty_list(int64_ndarray)
        self.values = typed.List.empty_list(float_ndarray)

    def add(self, term_ix, docids, values):
        self.term_ix.append(term_ix)
        self.docids.append(docids)
        self.values.append(values)


@numba.njit(parallel=True)
def encode_documents_numba(vectors: np.ndarray, docids: np.array, postings: Postings):
    #  -> List[Tuple[int, 'np.ndarray[int]', 'np.ndarray[float]']]:
    buffer = Buffer()

    for term_ix in numba.prange(vectors.shape[1]):
        v = vectors[:, term_ix]
        pos = postings.npostings[term_ix]
        (nonzero,) = np.nonzero(v)
        postings.npostings[term_ix] = (
            postings.npostings[term_ix] + len(nonzero)
        ) % postings.max_postings

        for docix in nonzero:
            # docix = docix.item()
            postings.docids[term_ix, pos] = docids[docix]
            postings.values[term_ix, pos] = v[docix]
            pos += 1

            if pos >= postings.max_postings:
                with numba.objmode():
                    buffer.add(
                        term_ix,
                        postings.docids[term_ix].copy(),
                        postings.values[term_ix].copy(),
                    )
                pos = 0

    return buffer
