"""Index for sparse models"""

from experimaestro.core.arguments import Constant
import torch
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Iterator, List, BinaryIO, NamedTuple, Sized, Tuple
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
import numba.extending
import operator

import numba
from numba.typed import List as NumbaList
from numba.experimental import jitclass


logger = easylog()

# Defines global types
DOCID_TYPE = np.uint32
DOCID_ITEMSIZE = np.dtype(DOCID_TYPE).itemsize
DOCID_TYPE_NUMBA = numba.uint32
DOCIDS_TYPE_NUMBA = DOCID_TYPE_NUMBA[:]

VALUE_TYPE = np.float32
VALUE_ITEMSIZE = np.dtype(VALUE_TYPE).itemsize
VALUE_TYPE_NUMBA = numba.float32
VALUES_TYPE_NUMBA = VALUE_TYPE_NUMBA[:]


@jitclass([("docid", DOCID_TYPE_NUMBA), ("value", VALUE_TYPE_NUMBA)])
class InternalScoredDocument:
    docid: int
    score: float

    def __init__(self, docid, score):
        self.docid = docid
        self.score = score

    def __lt__(self, other: "InternalScoredDocument"):
        return self.score < other.score


@jitclass(
    [("index", numba.int16), ("docid", DOCID_TYPE_NUMBA), ("value", VALUE_TYPE_NUMBA)]
)
class Posting:
    index: int
    """The iterator index"""

    docid: int
    """The document ID"""

    value: float
    """The value"""

    def __init__(self, index: int, docid: int, value: float):
        self.index = index
        self.docid = docid
        self.value = value

    def __lt__(self, other: "Posting"):
        return self.docid < other.docid


@numba.extending.overload(operator.lt)
def operator_lt(a, b):
    if Posting.class_type == a.class_type and Posting.class_type == b.class_type:

        def gt(a, b):
            return a.docid < b.docid

        return gt
    if (
        InternalScoredDocument.class_type == a.class_type
        and InternalScoredDocument.class_type == b.class_type
    ):

        def gt(a, b):
            return a.score < b.score

        return gt


@numba.extending.overload(operator.gt)
def operator_gt(a, b):
    if Posting.class_type == a.class_type and Posting.class_type == b.class_type:

        def gt(a, b):
            return a.docid > b.docid

        return gt
    if (
        InternalScoredDocument.class_type == a.class_type
        and InternalScoredDocument.class_type == b.class_type
    ):

        def gt(a, b):
            return a.score > b.score

        return gt


def read_postings(
    index_mm: mmap.mmap, start: int, length: int
) -> Tuple[np.ndarray, np.ndarray]:
    index_mm.seek(start)
    docids = np.frombuffer(index_mm.read(DOCID_ITEMSIZE * length), dtype=DOCID_TYPE)
    values = np.frombuffer(index_mm.read(VALUE_ITEMSIZE * length), dtype=VALUE_TYPE)
    return docids, values


@jitclass(
    [
        # Iterator index
        ("index", numba.int16),
        # document IDs and values
        ("docids", DOCIDS_TYPE_NUMBA),
        ("values", VALUES_TYPE_NUMBA),
        # Current iterator position
        ("ix", numba.int64),
    ]
)
class PostingIterator:
    def __init__(self, index: int, docids: np.ndarray, values: np.ndarray):
        self.index = index
        self.docids = docids
        self.values = values
        self.ix = 0

    def has_next(self):
        return self.ix < len(self.docids)

    def next(self):
        if not self.has_next():
            raise Exception("No more element in the interator")
        ix = self.ix
        self.ix += 1
        return Posting(self.index, int(self.docids[ix]), float(self.values[ix]))


# --- TAAT retrieval


@jitclass(
    [
        # Iterator index
        ("index", numba.int16),
        # document IDs and values
        ("docids", DOCIDS_TYPE_NUMBA),
        ("values", VALUES_TYPE_NUMBA),
        # Current iterator position
        ("ix", numba.int64),
        # Current maximum
        ("current_max_value", numba.float64),
        # Min value
        ("min_value", numba.float64),
    ]
)
class ImpactPostingIterator:
    def __init__(self, index: int, docids: np.ndarray, values: np.ndarray):
        self.index = index
        self.docids = docids
        self.values = values
        self.current_max_value = values.max()
        self.min_value = values.min()
        self.ix = 0

    def has_next(self):
        return self.ix < len(self.docids)

    def next(self):
        if not self.has_next():
            raise Exception("No more element in the interator")
        ix = self.ix
        self.ix += 1
        return Posting(self.index, int(self.docids[ix]), float(self.values[ix]))


@numba.njit(cache=True, parallel=False)
def retrieve_ordered(
    topk: int, query_values: List[float], iterators: List[ImpactPostingIterator]
):
    """Retrieve given that posting iterators are sorted by descending impact"""
    # Build a heap for iterators / postings
    pheap = [it.next() for it in iterators]
    heapq.heapify(pheap)

    # Document heap
    top = TopDocuments(topk)

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
        if iterators[tip.index].has_next():
            heapq.heapreplace(pheap, iterators[tip.index].next())
        else:
            heapq.heappop(pheap)

    # Add last document
    top.add(current)

    # Sort and returns
    top.dheap.sort(reverse=True)
    return top


# --- DAAT retrieval


@jitclass(
    [
        ("topk", numba.int32),
        (
            "dheap",
            numba.types.ListType(InternalScoredDocument.class_type.instance_type),
        ),
    ]
)
class TopDocuments:
    """Holder for top-K documents"""

    def __init__(self, topk: int):
        self.topk = topk
        self.dheap: List[InternalScoredDocument] = typed.List.empty_list(
            InternalScoredDocument(0, 0.0)
        )

    def add(self, current: InternalScoredDocument):
        if current.docid == -1:
            return

        if len(self.dheap) < self.topk:
            heapq.heappush(self.dheap, current)
        elif current.score > self.dheap[0].score:
            heapq.heapreplace(self.dheap, current)


@numba.njit(cache=True, parallel=False)
def retrieve(topk: int, query_values: List[float], iterators: List[PostingIterator]):
    # Build a heap for iterators / postings
    pheap = [it.next() for it in iterators]
    heapq.heapify(pheap)

    # Document heap
    top = TopDocuments(topk)

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
        if iterators[tip.index].has_next():
            heapq.heapreplace(pheap, iterators[tip.index].next())
        else:
            heapq.heappop(pheap)

    # Add last document
    top.add(current)

    # Sort and returns
    top.dheap.sort(reverse=True)
    return top


@numba.njit(cache=True, parallel=True)
def retrieve_all(
    topk: int,
    all_query_values: List[List[float]],
    all_iterators: List[List[PostingIterator]],
):
    results = NumbaList()

    for ix in numba.prange(len(all_query_values)):
        query_values = all_query_values[ix]
        iterators = all_iterators[ix]
        results.append((ix, retrieve(topk, query_values, iterators)))

    results.sort(key=lambda t: t[0])
    return [r[1] for r in results]


# --- Index and retriever


class SparseRetrieverIndex(Config):
    docids_path: Meta[Path]
    values_path: Meta[Path]
    info_path: Meta[Path]
    documents: Param[AdhocDocumentStore]
    ordered: Param[bool]

    starts: array.array
    lengths: array.array

    def initialize(self, in_memory: bool):
        with self.info_path.open("rb") as fp:
            self.starts = pickle.load(fp)

        if in_memory:
            self.values = np.fromfile(self.values_path, dtype=VALUE_TYPE)
            self.docids = np.fromfile(self.docids_path, dtype=DOCID_TYPE)
        else:
            self.values = np.memmap(self.values_path, mode="r", dtype=VALUE_TYPE)
            self.docids = np.memmap(self.docids_path, mode="r", dtype=DOCID_TYPE)

        numentries = self.starts[-1]

        assert len(self.values) == numentries, f"{len(self.values)} != {numentries}"
        assert len(self.docids) == numentries, f"{len(self.docids)} != {numentries}"

    def postings(self, index: int, term_ix: int) -> PostingIterator:
        start, end = self.starts[term_ix : (term_ix + 2)]
        pi = PostingIterator(index, self.docids[start:end], self.values[start:end])
        return pi


class SparseRetriever(Retriever):
    index: Param[SparseRetrieverIndex]
    encoder: Param[TextEncoder]
    topk: Param[int]

    batcher: Meta[Batcher] = Batcher()
    """The way to prepare batches of queries (when using retrieve_all)"""

    batchsize: Meta[int]
    """Size of batches (when using retrieve_all)"""

    in_memory: Meta[bool] = False
    """Whether the index should be fully loaded in memory (otherwise, uses virtual memory)"""

    def initialize(self):
        # TODO: initialize without parameters should be transparent
        super().initialize()
        self.encoder.initialize()
        self.index.initialize(self.in_memory)

    def retrieve_all(self, queries: Dict[str, str]) -> Dict[str, List[ScoredDocument]]:
        def reducer(
            batch: List[Tuple[str, str]], results: Dict[str, List[ScoredDocument]]
        ):
            encoded = self.encoder([text for _, text in batch]).cpu()
            all_query_values: List[List[float]] = NumbaList()
            all_iterators: List[List[PostingIterator]] = NumbaList()
            for vector in encoded:
                iterators, query_values = self.build_iterators(vector)
                all_query_values.append(query_values)
                all_iterators.append(iterators)

            tops = retrieve_all(self.topk, all_query_values, all_iterators)
            for (key, _), top in zip(batch, tops):
                results[key] = self.toptolist(top)
            return results

        batcher = self.batcher.initialize(self.batchsize)
        results = {}
        with tqdm(list(queries.items()), desc="Retrieve documents") as it:
            for batch in batchiter(self.batchsize, it):
                results = batcher.reduce(batch, reducer, results)

        return results

    def retrieve(self, query: str) -> List[ScoredDocument]:
        """Search with document-at-a-time (DAAT) strategy"""

        # Build up iterators
        vector = self.encoder([query])[0].cpu()
        iterators, query_values = self.build_iterators(vector)

        top = retrieve(self.topk, query_values, iterators)
        return self.toptolist(top)

    def build_iterators(self, vector: torch.Tensor):
        query_values: List[float] = NumbaList()
        iterators: List[PostingIterator] = NumbaList()

        iterator_index = 0
        for ix in torch.nonzero(vector):
            iterators.append(self.index.postings(iterator_index, int(ix)))
            query_values.append(float(vector[ix]))
            iterator_index += 1

        return iterators, query_values

    def toptolist(self, top: TopDocuments):
        return [
            ScoredDocument(
                self.index.documents.docid_internal2external(d.docid), d.score, None
            )
            for d in top.dheap
        ]


@jitclass(
    [
        ("max_postings", numba.int64),
        ("values", VALUE_TYPE_NUMBA[:, :]),
        ("docids", DOCID_TYPE_NUMBA[:, :]),
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

    ordered_index: Param[bool]
    """Ordered index: if not ordered, use DAAT strategy (WAND), otherwise, use fast top-k strategies"""

    device: Meta[Device] = DEFAULT_DEVICE

    max_postings: Meta[int] = 16384
    """Maximum number of postings (per term) before flushing to disk"""

    values_path: Annotated[Path, pathgenerator("values.bin")]
    docids_path: Annotated[Path, pathgenerator("docids.bin")]
    info_path: Annotated[Path, pathgenerator("info.bin")]

    version: Constant[int] = 3
    """Version 3 of the index"""

    def taskoutputs(self):
        """Returns a sparse retriever index that can be used by a SparseRetriever to search efficiently
        for documents"""
        return SparseRetrieverIndex(
            values_path=self.values_path,
            docids_path=self.docids_path,
            info_path=self.info_path,
            documents=self.documents,
            ordered=self.ordered_index,
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
        self.posting_starts = [array.array("L") for _ in range(self.encoder.dimension)]
        self.posting_lengths = [array.array("I") for _ in range(self.encoder.dimension)]

        logger.info(f"Starting to index {self.documents.documentcount} documents")

        path = self.values_path.parent / "temporary.bin"
        starts = array.array("L")
        try:
            with path.open("wb") as index_out:
                with torch.no_grad():
                    for batch in batchiter(self.batch_size, doc_iter):
                        batcher.process(batch, self.encode_documents, index_out)

                # Build the final index
                for term_ix in np.nonzero(self.postings.npostings)[0]:
                    self.flush(term_ix, index_out)

            logger.info("Rewriting the index file")

            with self.values_path.open("wb") as values_out, self.docids_path.open(
                "wb"
            ) as docids_out, path.open("r+b") as mm_file:
                index_mm = mmap.mmap(mm_file.fileno(), 0)

                start = 0
                starts.append(0)
                for ix, (_starts, _lengths) in enumerate(
                    zip(self.posting_starts, self.posting_lengths)
                ):
                    # Set start of term
                    length = sum(_lengths)
                    start += length
                    starts.append(start)

                    # Read and write
                    docids = np.ndarray((length,), dtype=DOCID_TYPE)
                    values = np.ndarray((length,), dtype=VALUE_TYPE)
                    offset = 0
                    for term_start, term_length in zip(_starts, _lengths):
                        _docids, _values = read_postings(
                            index_mm, term_start, term_length
                        )
                        docids[offset : offset + term_length] = _docids
                        values[offset : offset + term_length] = _values
                        offset += term_length

                    # Output to file
                    if self.ordered_index:
                        # Sort by decreasing value
                        sort_ix = np.argsort(values)[::-1]
                        docids = docids[sort_ix]
                        values = values[sort_ix]

                    docids.tofile(docids_out)
                    values.tofile(values_out)
        finally:
            path.unlink()

        logger.info("Dumping the index to %s", self.info_path)
        with self.info_path.open("wb") as fp:
            pickle.dump(
                starts,
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


@jitclass(
    [
        ("term_ix", numba.types.ListType(numba.uint64)),
        ("docids", numba.types.ListType(DOCIDS_TYPE_NUMBA)),
        ("values", numba.types.ListType(VALUES_TYPE_NUMBA)),
    ]
)
class Buffer:
    def __init__(self):
        self.term_ix = typed.List.empty_list(numba.uint64)
        self.docids = typed.List.empty_list(DOCIDS_TYPE_NUMBA)
        self.values = typed.List.empty_list(VALUES_TYPE_NUMBA)

    def add(self, term_ix, docids: np.ndarray, values: np.ndarray):
        self.term_ix.append(term_ix)
        self.docids.append(docids)
        self.values.append(values)


@numba.njit(cache=True, parallel=True)
def encode_documents_numba(vectors: np.ndarray, docids: np.ndarray, postings: Postings):
    buffer = Buffer()

    for term_ix in numba.prange(vectors.shape[1]):
        v = vectors[:, term_ix]
        pos = postings.npostings[term_ix]
        (nonzero,) = np.nonzero(v)
        postings.npostings[term_ix] = (
            postings.npostings[term_ix] + len(nonzero)
        ) % postings.max_postings

        for docix in nonzero:
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
