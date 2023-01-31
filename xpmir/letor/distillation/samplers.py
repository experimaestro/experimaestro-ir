from typing import Iterable, Iterator, NamedTuple, Optional, Tuple
from datamaestro.data import File
from experimaestro import Config, Meta, Param
from xpmir.letor.records import Query
from datamaestro_text.data.ir import AdhocDocumentStore
from xpmir.letor.samplers import Sampler
from xpmir.rankers import ScoredDocument
from xpmir.datasets.adapters import TextStore
import numpy as np

from xpmir.utils.iter import SerializableIterator, SkippingIterator


class PairwiseDistillationSample(NamedTuple):
    query: Query
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

    documentstore: Param[Optional[AdhocDocumentStore]]
    """The store for document texts if needed"""

    querystore: Param[Optional[TextStore]]
    """The store for query texts if needed"""

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        for sample in self.samples:
            if self.querystore is not None:
                sample.query.text = self.querystore[sample.query.id]
            if self.documentstore is not None:
                for d in sample.documents:
                    d.content = self.documentstore.document_text(d.docid)

            yield sample


class PairwiseDistillationSamplesTSV(PairwiseDistillationSamples, File):
    """A TSV file (Score 1, Score 2, Query, Document 1, Document 2)"""

    with_docid: Meta[bool]
    with_queryid: Meta[bool]

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        return self.iter()

    def iter(self) -> Iterator[PairwiseDistillationSample]:
        import csv

        with self.path.open("rt") as fp:
            for row in csv.reader(fp, delimiter="\t"):
                if self.with_queryid:
                    query = Query(row[2], None)
                else:
                    query = Query(None, row[2])

                if self.with_docid:
                    documents = (
                        ScoredDocument(row[3], float(row[0]), None),
                        ScoredDocument(row[4], float(row[1]), None),
                    )
                else:
                    documents = (
                        ScoredDocument(None, float(row[0]), row[3]),
                        ScoredDocument(None, float(row[1]), row[4]),
                    )

                yield PairwiseDistillationSample(query, documents)


class DistillationPairwiseSampler(Sampler):
    """Just loops over samples"""

    samples: Param[PairwiseDistillationSamples]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

    def pairwise_iter(self) -> SerializableIterator[PairwiseDistillationSample]:
        return SkippingIterator(iter(self.samples))
