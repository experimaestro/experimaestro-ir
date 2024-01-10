from abc import ABC, abstractmethod
from typing import Iterator, Optional, List, Any
from datamaestro_text.data.ir.base import Document, Topic
from experimaestro import Config, Param
import numpy as np
import datamaestro_text.data.ir.base as ir
from datamaestro_text.data.ir import DocumentStore
from xpmir.datasets.adapters import TextStore
from xpmir.letor.samplers import PairwiseSampler
from xpmir.letor.records import (
    PairwiseRecords,
    PairwiseRecord,
    TopicRecord,
    DocumentRecord,
)
from xpmir.utils.iter import (
    SerializableIterator,
    SkippingIterator,
    SerializableIteratorTransform,
)


class SampleTransform(Config, ABC):
    @abstractmethod
    def transform_topics(self, topics: List[ir.Topic]) -> Optional[List[ir.Topic]]:
        ...

    @abstractmethod
    def transform_documents(
        self, documents: List[ir.Document]
    ) -> Optional[List[ir.Document]]:
        ...


class SampleHydrator(SampleTransform):
    """Base class for document/topic hydrators"""

    documentstore: Param[Optional[DocumentStore]]
    """The store for document texts if needed"""

    querystore: Param[Optional[TextStore]]
    """The store for query texts if needed"""

    def transform_topics(self, topics: List[ir.Topic]):
        if self.querystore is None:
            return None
        return [
            ir.GenericTopic(topic.get_id(), self.querystore[topic.get_id()])
            for topic in topics
        ]

    def transform_documents(self, documents: List[ir.Document]):
        if self.documentstore is None:
            return None
        return self.documentstore.documents_ext([d.id for d in documents])


class SamplePrefixAdding(SampleTransform):
    """Transform the query and documents by adding the prefix"""

    query_prefix: Param[str] = ""
    """The prefix for the query"""

    document_prefix: Param[str] = ""
    """The prefix for the document"""

    def transform_topics(self, topics: List[Topic]) -> List[Topic] | None:
        if self.query_prefix == "" or len(topics) == 0:
            return None

        if isinstance(topics[0], ir.GenericTopic):
            return [
                ir.GenericTopic(topic.get_id(), self.query_prefix + topic.get_text())
                for topic in topics
            ]
        elif isinstance(topics[0], ir.TextTopic):
            return [
                ir.TextTopic(self.query_prefix + topic.get_text()) for topic in topics
            ]

    def transform_documents(self, documents: List[Document]) -> List[Document] | None:
        if self.document_prefix == "" or len(documents) == 0:
            return None

        if isinstance(documents[0], ir.GenericDocument):
            return [
                ir.GenericDocument(
                    document.get_id(), self.document_prefix + document.get_text()
                )
                for document in documents
            ]
        elif isinstance(documents[0], ir.TextDocument):
            return [
                ir.TextDocument(self.document_prefix + document.get_text())
                for document in documents
            ]


class PairwiseTransformAdapter(PairwiseSampler):
    """Transforms pairwise samples using an adapter

    It is interesting to use this adapter since the transformation is only
    performed if the samples are used: when using a SkippingIterator, when
    recovering a checkpoint, all the records might have to be processed
    otherwise.
    """

    sampler: Param[PairwiseSampler]
    """The distillation samples without texts for query and documents"""

    adapter: Param[SampleTransform]
    """The transformation"""

    def initialize(self, random: Optional[np.random.RandomState] = None):
        super().initialize(random)
        self.sampler.initialize(random)

    def transform_record(self, record: PairwiseRecord) -> PairwiseRecord:
        topics = [record.query.topic]
        docs = [record.positive.document, record.negative.document]

        topics = self.adapter.transform_topics(topics) or topics
        docs = self.adapter.transform_documents(docs) or docs

        return PairwiseRecord(
            TopicRecord(topics[0]), DocumentRecord(docs[0]), DocumentRecord(docs[1])
        )

    def pairwise_iter(self) -> Iterator[PairwiseRecord]:
        iterator = self.sampler.pairwise_iter()

        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform_record
        )

    def transform_records(self, records: PairwiseRecords) -> PairwiseRecords:
        for adapter in self.adapters:
            if topics := adapter.transform_topics(
                [tr.topic for tr in records.unique_topics]
            ):
                records.set_unique_topics([TopicRecord(topic) for topic in topics])

            if documents := adapter.transform_documents(
                [dr.document for dr in records.unique_documents]
            ):
                records.set_unique_documents(
                    [DocumentRecord(document) for document in documents]
                )
        return records

    def pairwise_batch_iter(self, size) -> SerializableIterator[PairwiseRecords, Any]:
        iterator = self.sampler.pairwise_batch_iter(size)
        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform_records
        )
