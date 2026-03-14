"""Processor hierarchy for transforming batches of samples.

Processors extract documents/queries from samples, process them in batch
(e.g., hydrate from a store), and put results back using the sample's
protocol methods (get_documents/with_documents, get_queries/with_queries).
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional

from experimaestro import Config, Param
from datamaestro_ir.data import DocumentStore, IDTextRecord, SimpleTextItem
from xpmir.datasets.adapters import TextStore
from xpmir.letor.records import SampleItem
from xpmir.rankers import ScoredDocument

DocIn = TypeVar("DocIn")
DocOut = TypeVar("DocOut")
QueryIn = TypeVar("QueryIn")
QueryOut = TypeVar("QueryOut")


class RecordsProcessor(Config, ABC, Generic[DocIn, QueryIn, DocOut, QueryOut]):
    """Processes a batch of SampleItem[DocIn, QueryIn] into
    SampleItem[DocOut, QueryOut]."""

    @abstractmethod
    def process_batch(
        self, records: List[SampleItem[DocIn, QueryIn]]
    ) -> List[SampleItem[DocOut, QueryOut]]: ...


class DocumentsProcessor(
    RecordsProcessor[DocIn, QueryIn, DocOut, QueryIn],
    Generic[DocIn, QueryIn, DocOut],
):
    """Extracts documents from samples, processes them in batch, puts them back.

    Queries are unchanged (QueryIn → QueryIn).
    """

    @abstractmethod
    def process_documents(self, documents: List[DocIn]) -> List[DocOut]: ...

    def process_batch(self, records):
        all_docs = []
        offsets = []
        for r in records:
            docs = r.get_documents()
            offsets.append(len(docs))
            all_docs.extend(docs)
        if not all_docs:
            return records
        processed = self.process_documents(all_docs)
        result = []
        idx = 0
        for r, n in zip(records, offsets):
            result.append(r.with_documents(processed[idx : idx + n]))
            idx += n
        return result


class QueriesProcessor(
    RecordsProcessor[DocIn, QueryIn, DocIn, QueryOut],
    Generic[DocIn, QueryIn, QueryOut],
):
    """Extracts queries from samples, processes them in batch, puts them back.

    Documents are unchanged (DocIn → DocIn).
    """

    @abstractmethod
    def process_queries(self, queries: List[QueryIn]) -> List[QueryOut]: ...

    def process_batch(self, records):
        all_queries = []
        offsets = []
        for r in records:
            qs = r.get_queries()
            offsets.append(len(qs))
            all_queries.extend(qs)
        if not all_queries:
            return records
        processed = self.process_queries(all_queries)
        result = []
        idx = 0
        for r, n in zip(records, offsets):
            result.append(r.with_queries(processed[idx : idx + n]))
            idx += n
        return result


class StoreHydrator(
    DocumentsProcessor[DocIn, QueryIn, DocOut],
    QueriesProcessor[DocIn, QueryIn, QueryOut],
):
    """Hydrates ID-only records with text from document/query stores.

    When documentstore is set, documents are hydrated via documents_ext().
    When querystore is set, queries are hydrated via store lookup.
    For documents with ScoredItem, the score is preserved via ScoredDocument.
    """

    documentstore: Param[Optional[DocumentStore]]
    querystore: Param[Optional[TextStore]]

    def process_documents(self, docs):
        if self.documentstore is None:
            return docs
        # Extract IDs, hydrate, preserve scores if present
        ids = [
            d.document["id"] if isinstance(d, ScoredDocument) else d["id"] for d in docs
        ]
        hydrated = self.documentstore.documents_ext(ids)
        result = []
        for orig, new_doc in zip(docs, hydrated):
            if isinstance(orig, ScoredDocument):
                result.append(ScoredDocument(new_doc, orig.score))
            else:
                result.append(new_doc)
        return result

    def process_queries(self, queries):
        if self.querystore is None:
            return queries
        return [
            IDTextRecord(id=q["id"], text_item=SimpleTextItem(self.querystore[q["id"]]))
            for q in queries
        ]

    def process_batch(self, records):
        records = QueriesProcessor.process_batch(self, records)
        records = DocumentsProcessor.process_batch(self, records)
        return records
