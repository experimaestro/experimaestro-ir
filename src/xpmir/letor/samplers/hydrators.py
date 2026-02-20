from abc import ABC, abstractmethod
from typing import Optional, List
from experimaestro import Config, Param
import datamaestro_text.data.ir.base as ir
from datamaestro_text.data.ir import DocumentStore, IDItem
from xpmir.datasets.adapters import TextStore


class SampleTransform(Config, ABC):
    @abstractmethod
    def transform_topics(
        self, topics: List[ir.TopicRecord]
    ) -> Optional[List[ir.TopicRecord]]: ...

    @abstractmethod
    def transform_documents(
        self, documents: List[ir.DocumentRecord]
    ) -> Optional[List[ir.DocumentRecord]]: ...


class SampleHydrator(SampleTransform):
    """Base class for document/topic hydrators (deprecated: use StoreHydrator + SamplerAdapter)"""

    documentstore: Param[Optional[DocumentStore]]
    """The store for document texts if needed"""

    querystore: Param[Optional[TextStore]]
    """The store for query texts if needed"""

    def transform_topics(self, topics: List[ir.TopicRecord]):
        if self.querystore is None:
            return None
        return [
            ir.create_record(
                id=topic[IDItem].id, text=self.querystore[topic[IDItem].id]
            )
            for topic in topics
        ]

    def transform_documents(self, documents: List[ir.DocumentRecord]):
        if self.documentstore is None:
            return None
        return self.documentstore.documents_ext([d[IDItem].id for d in documents])


class SamplePrefixAdding(SampleTransform):
    """Transform the query and documents by adding the prefix"""

    query_prefix: Param[str] = ""
    """The prefix for the query"""

    document_prefix: Param[str] = ""
    """The prefix for the document"""

    def transform_topics(
        self, topics: List[ir.TopicRecord]
    ) -> Optional[List[ir.TopicRecord]]:
        if self.query_prefix == "" or len(topics) == 0:
            return None

        if isinstance(topics[0], ir.GenericTopic):
            return [
                ir.GenericTopic(topic[IDItem].id, self.query_prefix + topic.text)
                for topic in topics
            ]
        elif isinstance(topics[0], ir.TextTopic):
            return [ir.TextTopic(self.query_prefix + topic.text) for topic in topics]

    def transform_documents(
        self, documents: List[ir.DocumentRecord]
    ) -> Optional[List[ir.DocumentRecord]]:
        if self.document_prefix == "" or len(documents) == 0:
            return None

        if isinstance(documents[0], ir.GenericDocument):
            return [
                ir.GenericDocument(
                    document[IDItem].id, self.document_prefix + document.text
                )
                for document in documents
            ]
        elif isinstance(documents[0], ir.TextDocument):
            return [
                ir.TextDocument(self.document_prefix + document.text)
                for document in documents
            ]


class SampleTransformList(SampleTransform):
    """A class which group a list of sample transforms"""

    adapters: Param[List[SampleTransform]]
    """The list of sample transform to be applied"""

    def transform_topics(self, topics: List[ir.TopicRecord]) -> List[ir.TopicRecord]:
        for adapter in self.adapters:
            topics = adapter.transform_topics(topics) or topics
        return topics

    def transform_documents(
        self, documents: List[ir.DocumentRecord]
    ) -> List[ir.DocumentRecord]:
        for adapter in self.adapters:
            documents = adapter.transform_documents(documents) or documents
        return documents
