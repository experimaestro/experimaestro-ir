"""Bag-of-Words index with BM25 scoring using impact_index"""

import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
from datamaestro_ir.data import DocumentStore, IDTextRecord

from xpmir.rankers import Retriever, ScoredDocument
from xpmir.rankers.standard import Model, BM25
import impact_index

logger = logging.getLogger(__name__)


class BOWSparseRetrieverIndex(Config):
    """A bag-of-words index with BM25 scoring

    Uses impact_index.BOWIndexBuilder for text-based tokenization
    and BM25 scoring at retrieval time.
    """

    documents: Param[DocumentStore]
    """The indexed document collection"""

    index_path: Meta[Path]
    """Path to the index directory"""

    stemmer: Param[Optional[str]] = field(default=None, ignore_default=True)
    """Stemmer to use (e.g. 'snowball')"""

    language: Param[Optional[str]] = field(default=None, ignore_default=True)
    """Language for stemming (e.g. 'english')"""

    def initialize(self, in_memory: bool, model: Model):
        """Initialize the index with scoring model

        :param in_memory: Whether to load the index fully in memory
        :param model: The scoring model (BM25)
        """
        index = impact_index.Index.load(str(self.index_path.absolute()), in_memory)
        doc_meta = impact_index.DocMetadata.load(str(self.index_path.absolute()))

        if isinstance(model, BM25):
            scoring = impact_index.BM25Scoring(k1=model.k1, b=model.b)
        else:
            raise ValueError(f"Unsupported model type: {type(model)}")

        self.scored_index = index.with_scoring(scoring, doc_meta)

        # Load analyzer for query tokenization
        self._analyzer = impact_index.TextAnalyzer.load(
            str(self.index_path.absolute()),
            stemmer=self.stemmer,
            language=self.language,
        )

    def analyze_query(self, text: str) -> Dict[int, float]:
        """Tokenize and stem a query string into term IDs"""
        return self._analyzer.analyze_query(text)

    def retrieve(self, query: Dict[int, float], top_k: int) -> List[ScoredDocument]:
        results = []
        for sd in self.scored_index.search_wand(query, top_k):
            results.append(
                ScoredDocument(
                    self.documents.document_int(sd.docid),
                    sd.score,
                )
            )
        return results


class BOWRetriever(Retriever):
    """BM25 retriever using the impact_index BOW index

    This mirrors the AnseriniRetriever but uses the impact_index library
    for BM25 scoring instead of Lucene/pyserini.
    """

    index: Param[BOWSparseRetrieverIndex]
    """The BOW index"""

    model: Param[Model]
    """The scoring model (e.g. BM25)"""

    topk: Param[int]
    """Number of documents to return"""

    in_memory: Meta[bool] = field(default=False, ignore_default=True)
    """Whether the index should be fully loaded in memory"""

    def initialize(self):
        super().initialize()
        self.index.initialize(self.in_memory, self.model)

    def retrieve(self, record: IDTextRecord) -> List[ScoredDocument]:
        text = record["text_item"].text if isinstance(record, dict) else record
        query = self.index.analyze_query(text)
        return self.index.retrieve(query, self.topk)

    def retrieve_all(
        self, queries: Dict[str, IDTextRecord]
    ) -> Dict[str, List[ScoredDocument]]:
        results = {}
        for key, record in tqdm(list(queries.items())):
            results[key] = self.retrieve(record)
        return results


class BOWSparseRetrieverIndexBuilder(Task):
    """Builds a bag-of-words index from document text

    Uses impact_index.BOWIndexBuilder to tokenize documents and store
    term frequencies + document lengths for BM25 scoring.
    """

    documents: Param[DocumentStore]
    """Set of documents to index"""

    stemmer: Param[Optional[str]] = field(default=None, ignore_default=True)
    """Stemmer to use (e.g. 'snowball')"""

    language: Param[Optional[str]] = field(default=None, ignore_default=True)
    """Language for stemming (e.g. 'english')"""

    batch_size: Param[int] = field(default=1000, ignore_default=True)
    """Batch size for progress reporting"""

    max_docs: Param[int] = field(default=0, ignore_default=True)
    """Maximum number of indexed documents (0 = all)"""

    index_path: Meta[Path] = field(default_factory=PathGenerator("index"))
    """Path to store the index"""

    version: Constant[int] = 1
    """Version of the BOW index"""

    def execute(self):
        if self.index_path.is_dir():
            shutil.rmtree(self.index_path)
        self.index_path.mkdir(parents=True, exist_ok=True)

        kwargs = {"dtype": "int32"}
        if self.stemmer is not None:
            kwargs["stemmer"] = self.stemmer
        if self.language is not None:
            kwargs["language"] = self.language

        builder = impact_index.BOWIndexBuilder(str(self.index_path), **kwargs)

        max_docs = self.documents.documentcount
        if self.max_docs:
            max_docs = min(self.max_docs, max_docs or sys.maxsize)
            logger.warning("Limited indexing to %d documents", max_docs)

        with tqdm(
            total=max_docs,
            unit="documents",
            desc="Building BOW index",
        ) as pb:
            for docid, doc in enumerate(self.documents.iter_documents()):
                if self.max_docs and docid >= max_docs:
                    break
                text = doc["text_item"].text
                builder.add_text(docid, text)
                pb.update()

        logger.info("Building the index")
        builder.build(False)
        logger.info("BOW index built")

    def task_outputs(self, dep):
        """Returns a BOW index that can be used by a BOWRetriever"""
        kwargs = {}
        if self.stemmer is not None:
            kwargs["stemmer"] = self.stemmer
        if self.language is not None:
            kwargs["language"] = self.language

        return dep(
            BOWSparseRetrieverIndex.C(
                index_path=self.index_path,
                documents=self.documents,
                **kwargs,
            )
        )
