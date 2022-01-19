"""Index for sparse models"""

import torch
from pathlib import Path
from typing import Iterator, List, BinaryIO, NamedTuple, Tuple
from experimaestro import Annotated, Config, Task, Param, Meta, pathgenerator, tqdm
from datamaestro_text.data.ir import AdhocDocument, AdhocDocumentStore
from xpmir.letor.batchers import Batcher
from xpmir.utils import batchiter
from xpmir.letor import Device, DEFAULT_DEVICE
from xpmir.text.encoders import TextEncoder
from xpmir.rankers import Retriever, ScoredDocument
import array
import pickle
import mmap
import heapq


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
    docid: int
    value: float

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
            pickle.load(fp)
            self.starts, self.lengths = pickle.load(fp)

        self.index_mm = mmap.mmap(self.index_path.open("r+b").fileno(), 0)

    def iter_postings(self, ix: int) -> Iterator[Posting]:
        for start, length in zip(self.starts[ix], self.lengths[ix]):
            # Read from mmap
            self.index_mm.seek(start)
            docids, values = docidArray(), valuesArray()
            docids.frombytes(self.index_mm.read(docids.itemsize * length))
            values.frombytes(self.index_mm.read(values.itemsize * length))

            # Output postings
            for docid, value in zip(docids, values):
                yield Posting(ix, docid, value)


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
        iterators: List[Iterator[Posting]] = []
        for ix, value in enumerate(vector):
            if value != 0:
                iterators.append(self.index.iter_postings(ix))

        # Build a heap for iterators / postings
        pheap = [next(iter) for iter in iterators]
        heapq.heapify(pheap)

        # Document heap
        top = TopDocuments(self.topk)

        # While we have posting to process
        current = InternalScoredDocument(-1, 0)
        while pheap:
            tip = pheap[0]
            if tip.docid == current.docid:
                current.score += tip.value
            else:
                top.add(current)
                current = InternalScoredDocument(tip.docid, tip.value)

            # Fetch next
            try:
                heapq.heapreplace(pheap, next(iterators[tip.index]))
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
    encoder: Param[TextEncoder]

    batcher: Meta[Batcher] = Batcher()
    batch_size: Param[int]
    threshold: Param[float] = 0

    device: Meta[Device] = DEFAULT_DEVICE

    max_postings: Meta[int] = 16384
    """Maximum number of postings (per term) before flushing to disk"""

    index_path: Annotated[Path, pathgenerator("index.bin")]
    info_path: Annotated[Path, pathgenerator("info.bin")]

    def config(self):
        return SparseRetrieverIndex(
            index_path=self.index_path,
            info_path=self.info_path,
            documents=self.documents,
        )

    def __init__(self):
        #: Holds the full list of document IDs
        self.docids = []
        self.internal_docid = 0

        # In memory index
        self.termsinfo: List[TermInfo] = []
        self.postings = 0

        self.index_pos = 0

    def execute(self):
        # Encode all documents
        batcher = self.batcher.initialize(self.batch_size)

        doc_iter = tqdm(
            self.documents.iter_documents(), total=self.documents.documentcount
        )

        with self.index_path.open("wb") as index_out:
            self.encoder.to(self.device()).eval()
            with torch.no_grad():
                for batch in batchiter(self.batch_size, doc_iter):
                    batcher.process(batch, self.encode_documents, index_out)

            # Build the final index
            for ti in self.termsinfo:
                if len(ti.docids) > 0:
                    self.flush(ti, index_out)

        with self.info_path.open("wb") as fp:
            pickle.dump(
                (
                    self.docids,
                    [ti.starts for ti in self.termsinfo],
                    [ti.lengths for ti in self.termsinfo],
                ),
                fp,
            )

    def encode_documents(self, batch: List[AdhocDocument], index_out: BinaryIO):
        # Assumes for now dense vectors
        self.internal_docid = 0
        vectors = self.encoder([d.text for d in batch]).cpu()

        for docid, v in zip((d.docid for d in batch), vectors):
            self.docids.append(docid)
            vabs = v.abs()
            for i in range(len(v)):
                if vabs[i] > 0:
                    while len(self.termsinfo) < i:
                        self.termsinfo.append(TermInfo())
                    ti = self.termsinfo[i]
                    ti.docids.append(self.internal_docid)
                    ti.values.append(v[i])
                    if len(ti.docids) >= self.max_postings:
                        self.flush(ti, index_out)

            self.internal_docid += 1

    def flush(self, ti: TermInfo, index_out: BinaryIO):
        """Flush term postings"""
        ti.starts.append(self.index_pos)
        ti.lengths.append(len(ti.docids))
        ti.docids.tofile(index_out)
        ti.values.tofile(index_out)
        self.index_pos += len(ti.docids)
        ti.reset()
