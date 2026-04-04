from abc import ABC, abstractmethod
from typing import Optional, List
from experimaestro import field, Config, Param
import datamaestro_ir.data.base as ir
from datamaestro_ir.data import DocumentStore, SimpleTextItem
from xpmir.datasets.adapters import TextStore


class SampleTransform(Config, ABC):
    @abstractmethod
    def transform_topics(
        self, topics: List[ir.IDTextRecord]
    ) -> Optional[List[ir.IDTextRecord]]: ...

    @abstractmethod
    def transform_documents(
        self, documents: List[ir.IDTextRecord]
    ) -> Optional[List[ir.IDTextRecord]]: ...


class SampleHydrator(SampleTransform):
    """Base class for document/topic hydrators (deprecated: use StoreHydrator + SamplerAdapter)"""

    documentstore: Param[Optional[DocumentStore]]
    """The store for document texts if needed"""

    querystore: Param[Optional[TextStore]]
    """The store for query texts if needed"""

    def transform_topics(self, topics: List[ir.IDTextRecord]):
        if self.querystore is None:
            return None
        return [
            {
                "id": topic["id"],
                "text_item": SimpleTextItem(self.querystore[topic["id"]]),
            }
            for topic in topics
        ]

    def transform_documents(self, documents: List[ir.IDTextRecord]):
        if self.documentstore is None:
            return None
        return self.documentstore.documents_ext([d["id"] for d in documents])


class SamplePrefixAdding(SampleTransform):
    """Transform the query and documents by adding the prefix"""

    query_prefix: Param[str] = field(default="", ignore_default=True)
    """The prefix for the query"""

    document_prefix: Param[str] = field(default="", ignore_default=True)
    """The prefix for the document"""

    def transform_topics(
        self, topics: List[ir.IDTextRecord]
    ) -> Optional[List[ir.IDTextRecord]]:
        if self.query_prefix == "" or len(topics) == 0:
            return None

        return [
            {
                **topic,
                "text_item": SimpleTextItem(
                    self.query_prefix + topic["text_item"].text
                ),
            }
            for topic in topics
        ]

    def transform_documents(
        self, documents: List[ir.IDTextRecord]
    ) -> Optional[List[ir.IDTextRecord]]:
        if self.document_prefix == "" or len(documents) == 0:
            return None

        return [
            {
                **doc,
                "text_item": SimpleTextItem(
                    self.document_prefix + doc["text_item"].text
                ),
            }
            for doc in documents
        ]


class SampleTransformList(SampleTransform):
    """A class which group a list of sample transforms"""

    adapters: Param[List[SampleTransform]]
    """The list of sample transform to be applied"""

    def transform_topics(self, topics: List[ir.IDTextRecord]) -> List[ir.IDTextRecord]:
        for adapter in self.adapters:
            topics = adapter.transform_topics(topics) or topics
        return topics

    def transform_documents(
        self, documents: List[ir.IDTextRecord]
    ) -> List[ir.IDTextRecord]:
        for adapter in self.adapters:
            documents = adapter.transform_documents(documents) or documents
        return documents
