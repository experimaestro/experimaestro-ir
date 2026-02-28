from abc import ABC, abstractmethod
from typing import Iterator, Optional, List, Any
from experimaestro import Config, Param
import numpy as np
import datamaestro_text.data.ir.base as ir
from datamaestro_text.data.ir import DocumentStore, SimpleTextItem
from xpmir.datasets.adapters import TextStore
from xpmir.letor.samplers import PairwiseSampler
from xpmir.letor.records import (
    PairwiseRecords,
    PairwiseRecord,
)
from xpmir.utils.iter import (
    SerializableIterator,
    SkippingIterator,
    SerializableIteratorTransform,
)


class SampleTransform(Config, ABC):
    @abstractmethod
    def transform_topics(
        self, topics: List[ir.IDTextRecord]
    ) -> Optional[List[ir.IDTextRecord]]:
        ...

    @abstractmethod
    def transform_documents(
        self, documents: List[ir.IDTextRecord]
    ) -> Optional[List[ir.IDTextRecord]]:
        ...


class SampleHydrator(SampleTransform):
    """Base class for document/topic hydrators"""

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

    query_prefix: Param[str] = ""
    """The prefix for the query"""

    document_prefix: Param[str] = ""
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
        topics = [record.query]
        docs = [record.positive, record.negative]

        topics = self.adapter.transform_topics(topics) or topics
        docs = self.adapter.transform_documents(docs) or docs

        return PairwiseRecord(topics[0], docs[0], docs[1])

    def pairwise_iter(self) -> Iterator[PairwiseRecord]:
        iterator = self.sampler.pairwise_iter()

        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform_record
        )

    def transform_records(self, records: PairwiseRecords) -> PairwiseRecords:
        if topics := self.adapter.transform_topics(
            topic for topic in records.unique_topics
        ):
            records.set_unique_topics(topics)

        if documents := self.adapter.transform_documents(
            document for document in records.unique_documents
        ):
            records.set_unique_documents(documents)
        return records

    def pairwise_batch_iter(self, size) -> SerializableIterator[PairwiseRecords, Any]:
        iterator = self.sampler.pairwise_batch_iter(size)
        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform_records
        )
