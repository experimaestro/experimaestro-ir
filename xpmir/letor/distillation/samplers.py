from typing import Dict, Iterable, Iterator, NamedTuple, Optional, Tuple
from datamaestro.data import File
from experimaestro import Config, Meta, Param
from xpmir.letor.records import Query
from datamaestro_text.data.ir import AdhocDocumentStore
from xpmir.letor.samplers import Sampler
from xpmir.rankers import ScoredDocument
from xpmir.datasets.adapters import TextStore
import numpy as np


class PairwiseDistillationSample(NamedTuple):
    query: Query
    documents: Tuple[ScoredDocument, ScoredDocument]


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
        self.count = 0
        super().initialize(random)

    # def __init__(self):
    #     self.count = 0

    def state_dict(self) -> Dict:
        return {"count": self.count}

    def load_state_dict(self, state: Dict):
        self.count = state["count"]

    def pairwise_iter(self) -> Iterator[PairwiseDistillationSample]:
        base = self.count
        while True:
            self.count = 0
            for sample in self.samples:
                if self.count >= base:
                    yield sample
                self.count += 1
