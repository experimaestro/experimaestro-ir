from typing import (
    Iterable,
    Iterator,
    NamedTuple,
    Tuple,
    List,
    Any,
)

import numpy as np
from datamaestro.data import File
from datamaestro_text.data.ir.base import (
    TopicRecord,
    DocumentRecord,
    ScoredItem,
    SimpleTextItem,
    IDItem,
    create_record,
)
from experimaestro import Config, Meta, Param
from xpmir.learning import Sampler
from xpmir.letor.samplers.hydrators import SampleHydrator
from xpmir.rankers import ScoredDocument
from xpmir.utils.iter import (
    SerializableIterator,
    SkippingIterator,
    SerializableIteratorTransform,
)


class PairwiseDistillationSample(NamedTuple):
    query: TopicRecord
    """The query"""

    documents: Tuple[DocumentRecord, DocumentRecord]
    """Positive/negative document with teacher scores"""


class PairwiseDistillationSamples(Config, Iterable[PairwiseDistillationSample]):
    """Pairwise distillation file"""

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        raise NotImplementedError()


class PairwiseHydrator(PairwiseDistillationSamples, SampleHydrator):
    """Hydrate ID-based samples with document and/or query content"""

    samples: Param[PairwiseDistillationSamples]
    """The distillation samples without texts for query and documents"""

    def transform(self, sample: PairwiseDistillationSample):
        topic, documents = sample.query, sample.documents

        if transformed := self.transform_topics(topic):
            topic = transformed[0]

        if transformed := self.transform_documents(documents):
            documents = tuple(
                ScoredDocument(d, sd[ScoredItem].score)
                for d, sd in zip(transformed, sample.documents)
            )

        return PairwiseDistillationSample(topic, documents)

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        iterator = iter(self.samples)
        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform
        )


class PairwiseDistillationSamplesTSV(PairwiseDistillationSamples, File):
    """A TSV file (Score 1, Score 2, Query, Document 1, Document 2)"""

    with_docid: Meta[bool]
    with_queryid: Meta[bool]

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        return self.iter()

    def iter(self) -> Iterator[PairwiseDistillationSample]:
        import csv

        def iterate():
            with self.path.open("rt") as fp:
                for row in csv.reader(fp, delimiter="\t"):
                    if self.with_queryid:
                        query = create_record(id=row[2])
                    else:
                        query = create_record(text=row[2])

                    if self.with_docid:
                        documents = (
                            DocumentRecord(IDItem(row[3]), ScoredItem(float(row[0]))),
                            DocumentRecord(IDItem(row[4]), ScoredItem(float(row[1]))),
                        )
                    else:
                        documents = (
                            DocumentRecord(
                                SimpleTextItem(row[3]), ScoredItem(float(row[0]))
                            ),
                            DocumentRecord(
                                SimpleTextItem(row[4]), ScoredItem(float(row[1]))
                            ),
                        )

                    yield PairwiseDistillationSample(query, documents)

        return SkippingIterator(iterate())


class DistillationPairwiseSampler(Sampler):
    """Just loops over samples"""

    samples: Param[PairwiseDistillationSamples]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

    def pairwise_iter(self) -> SerializableIterator[PairwiseDistillationSample, Any]:
        return SkippingIterator.make_serializable(iter(self.samples))

    def pairwise_batch_iter(
        self, size
    ) -> SerializableIterator[List[PairwiseDistillationSample], Any]:
        """Batchwise iterator

        Can be subclassed by some classes to be more efficient"""

        class BatchIterator(SerializableIterator):
            def __init__(self, sampler: DistillationPairwiseSampler):
                self.iter = sampler.pairwise_iter()

            def state_dict(self):
                return self.iter.state_dict()

            def load_state_dict(self, state):
                self.iter.load_state_dict(state)

            def __next__(self):
                batch = []
                for _, record in zip(range(size), self.iter):
                    batch.append(record)
                return batch

        return BatchIterator(self)
