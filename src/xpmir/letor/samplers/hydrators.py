from typing import Iterator, Optional, List, Any
from experimaestro import Config, Param

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


class SampleTransform(Config):
    pass


class SampleHydrator(SampleTransform):
    """Base class for document/topic hydrators"""

    documentstore: Param[Optional[DocumentStore]]
    """The store for document texts if needed"""

    querystore: Param[Optional[TextStore]]
    """The store for query texts if needed"""

    def transform_topics(self, topics: List[ir.Topic]):
        if self.querystore is None:
            return None
        return (
            ir.GenericTopic(topic.get_id(), self.querystore[topic.get_id()])
            for topic in topics
        )

    def transform_documents(self, documents: List[ir.Document]):
        if self.documentstore is None:
            return None
        return self.documentstore.documents_ext([d.id for d in documents])


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

    def transform_record(self, record: PairwiseRecord) -> PairwiseRecord:
        (topic,) = self.adapter.transform_topics([record.query.topic])
        pos, neg = self.adapter.transform_documents(
            [record.positive.document, record.negative.document]
        )
        return PairwiseRecord(
            TopicRecord(topic), DocumentRecord(pos), DocumentRecord(neg)
        )

    def pairwise_iter(self) -> Iterator[PairwiseRecord]:
        iterator = self.sampler.pairwise_iter()

        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform_record
        )

    def transform_records(self, records: PairwiseRecords) -> PairwiseRecord:
        if topics := self.adapter.transform_topics(
            [tr.topic for tr in records.unique_topics]
        ):
            records.set_unique_topics([TopicRecord(topic) for topic in topics])

        if documents := self.adapter.transform_documents(
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
