from typing import Iterable, Iterator, NamedTuple, Optional, Tuple, List, Any

import numpy as np
from datamaestro.data import File
from datamaestro_text.data.ir import DocumentStore
from datamaestro_text.data.ir.base import (
    GenericTopic,
    IDTopic,
    TextTopic,
    IDDocument,
    TextDocument,
)
from experimaestro import Config, Meta, Param

from xpmir.datasets.adapters import TextStore
from xpmir.learning import Sampler
from xpmir.letor.records import TopicRecord
from xpmir.rankers import ScoredDocument
from xpmir.utils.iter import (
    SerializableIterator,
    SkippingIterator,
    SerializableIteratorTransform,
)


class PairwiseDistillationSample(NamedTuple):
    query: TopicRecord
    """The query"""

    documents: Tuple[ScoredDocument, ScoredDocument]
    """Positive/negative document with teacher scores"""


class PairwiseDistillationSamples(Config, Iterable[PairwiseDistillationSample]):
    """Pairwise distillation file"""

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        raise NotImplementedError()


class PairwiseHydrator(PairwiseDistillationSamples):
    """Hydrate ID-based samples with document and/or query content"""

    samples: Param[PairwiseDistillationSamples]
    """The distillation samples without texts for query and documents"""

    documentstore: Param[Optional[DocumentStore]]
    """The store for document texts if needed"""

    querystore: Param[Optional[TextStore]]
    """The store for query texts if needed"""

    def transform(self, sample):
        topic, documents = sample.query, sample.documents

        if self.querystore is not None:
            topic = GenericTopic(
                sample.query.get_id(), self.querystore[sample.query.get_id()]
            )
        if self.documentstore is not None:
            documents = tuple(
                ScoredDocument(
                    self.documentstore.document_ext(d.document.get_id()), d.score
                )
                for d in sample.documents
            )

        sample = PairwiseDistillationSample(topic, documents)
        return sample

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
                        query = IDTopic(row[2])
                    else:
                        query = TextTopic(row[2])

                    if self.with_docid:
                        documents = (
                            ScoredDocument(IDDocument(row[3]), float(row[0])),
                            ScoredDocument(IDDocument(row[4]), float(row[1])),
                        )
                    else:
                        documents = (
                            ScoredDocument(TextDocument(row[3]), float(row[0])),
                            ScoredDocument(TextDocument(row[4]), float(row[1])),
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

        class BatchIterator:
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
