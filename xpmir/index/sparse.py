"""Index for sparse models"""

import torch
from pathlib import Path
from typing import Iterator, List, BinaryIO, NamedTuple, Tuple
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
            docids, values = docidArray(), valuesArray()
            docids.frombytes(self.index_mm.read(docids.itemsize * length))
            values.frombytes(self.index_mm.read(values.itemsize * length))

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


class TermInfo:
    """In memory term information"""

    def __init__(self):
        self.starts = array.array("I")
        self.lengths = array.array("I")
        self.reset()

    def reset(self):
        self.docids = docidArray()
        self.values = valuesArray()


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

    def __init__(self):
        # In memory index
        self.termsinfo: List[TermInfo] = []
        self.postings = 0

        self.index_pos = 0

    def execute(self):
        # Encode all documents
        self.encoder.initialize()
        batcher = self.batcher.initialize(self.batch_size)

        doc_iter = tqdm(
            self.documents.iter_documents(),
            total=self.documents.documentcount,
            desc="Building the index",
        )

        with self.index_path.open("wb") as index_out:
            self.encoder.to(self.device.value).eval()
            with torch.no_grad():
                for batch in batchiter(self.batch_size, doc_iter):
                    batcher.process(batch, self.encode_documents, index_out)

            # Build the final index
            for ti in self.termsinfo:
                if len(ti.docids) > 0:
                    self.flush(ti, index_out)

        logger.info("Dumping the index to %s", self.info_path)
        with self.info_path.open("wb") as fp:
            pickle.dump(
                (
                    [ti.starts for ti in self.termsinfo],
                    [ti.lengths for ti in self.termsinfo],
                ),
                fp,
            )

    def encode_documents(self, batch: List[AdhocDocument], index_out: BinaryIO):
        # Assumes for now dense vectors
        vectors = self.encoder([d.text for d in batch]).cpu()

        for docid, v in zip((d.internal_docid for d in batch), vectors):
            assert docid is not None
            for ix in torch.nonzero(v):
                ix = int(ix)
                while len(self.termsinfo) < ix + 1:
                    self.termsinfo.append(TermInfo())
                ti = self.termsinfo[ix]
                ti.docids.append(docid)
                ti.values.append(v[ix])
                if len(ti.docids) >= self.max_postings:
                    self.flush(ti, index_out)

    def flush(self, ti: TermInfo, index_out: BinaryIO):
        """Flush term postings"""
        ti.starts.append(self.index_pos)
        ti.lengths.append(len(ti.docids))
        ti.docids.tofile(index_out)
        ti.values.tofile(index_out)
        self.index_pos += (
            len(ti.docids) * ti.docids.itemsize + len(ti.values) * ti.values.itemsize
        )
        ti.reset()
