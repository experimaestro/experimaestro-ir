"""Index for sparse models"""

import torch
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
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
from datamaestro_text.data.ir import Document, DocumentStore
from xpmir.learning.batchers import Batcher
from xpmir.utils.utils import batchiter, easylog
from xpmir.letor import Device, DEFAULT_DEVICE
from xpmir.text.encoders import TextEncoder
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.utils.iter import MultiprocessIterator
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


class SparseRetriever(Retriever):
    index: Param[SparseRetrieverIndex]
    encoder: Param[TextEncoder]
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
        self.encoder.initialize()
        self.index.initialize(self.in_memory)

    def retrieve_all(self, queries: Dict[str, str]) -> Dict[str, List[ScoredDocument]]:
        """Input queries: {id: text}"""

        def reducer(
            batch: List[Tuple[str, str]],
            results: Dict[str, List[ScoredDocument]],
            progress,
        ):
            for (key, _), vector in zip(
                batch, self.encoder([text for _, text in batch]).cpu().detach().numpy()
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

    def retrieve(self, query: str, top_k=None) -> List[ScoredDocument]:
        """Search with document-at-a-time (DAAT) strategy

        :param top_k: Overrides the default top-K value
        """

        # Build up iterators
        vector = self.encoder([query])[0].cpu().detach().numpy()
        (ix,) = vector.nonzero()  # ix represents the position without 0 in the vector
        query = {
            ix: float(v) for ix, v in zip(ix, vector[ix])
        }  # generate a dict: {position:value}
        return self.index.retrieve(query, top_k or self.topk)


class SparseRetrieverIndexBuilder(Task):
    """Builds an index from a sparse representation

    Assumes that document and queries have the same dimension, and
    that the score is computed through an inner product
    """

    documents: Param[DocumentStore]
    """Set of documents to index"""

    encoder: Param[TextEncoder]
    """The encoder"""

    batcher: Meta[Batcher] = Batcher()
    """Batcher used when computing representations"""

    batch_size: Param[int]
    """Size of batches"""

    ordered_index: Param[bool]
    """Ordered index: if not ordered, use DAAT strategy (WAND), otherwise, use
    fast top-k strategies"""

    device: Meta[Device] = DEFAULT_DEVICE

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
        # Encode all documents
        logger.info(
            f"Load the encoder and transfer to the target device {self.device.value}"
        )

        self.encoder.initialize()
        self.encoder.to(self.device.value).eval()

        batcher = self.batcher.initialize(self.batch_size)

        doc_iter = tqdm(
            zip(
                range(sys.maxsize if self.max_docs == 0 else self.max_docs),
                MultiprocessIterator(self.documents.iter_documents()),
            ),
            total=self.documents.documentcount
            if self.max_docs == 0
            else min(self.max_docs, self.documents.documentcount),
            desc="Building the index",
        )

        # Create the index builder
        from shutil import rmtree
        import xpmir_rust

        if self.index_path.is_dir():
            rmtree(self.index_path)
        self.index_path.mkdir(parents=True)

        self.indexer = xpmir_rust.index.SparseIndexer(str(self.index_path))

        # Index
        logger.info(f"Starting to index {self.documents.documentcount} documents")

        with torch.no_grad():
            for batch in batchiter(self.batch_size, doc_iter):
                batcher.process(batch, self.encode_documents)

        # Build the index
        self.indexer.build(self.in_memory)

    def encode_documents(self, batch: List[Tuple[int, Document]]):
        # Assumes for now dense vectors
        vectors = (
            self.encoder([d.get_text() for _, d in batch]).cpu().numpy()
        )  # bs * vocab
        for vector, (docid, _) in zip(vectors, batch):
            (nonzero_ix,) = vector.nonzero()
            self.indexer.add(docid, nonzero_ix.astype(np.uint64), vector[nonzero_ix])
