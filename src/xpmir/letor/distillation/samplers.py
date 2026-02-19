import logging
from typing import (
    Iterable,
    Iterator,
    NamedTuple,
    Tuple,
    List,
)
import numpy as np

from experimaestro import Config, Meta, Param
from datamaestro.data import File
from datamaestro_text.data.ir.base import (
    TopicRecord,
    DocumentRecord,
    ScoredItem,
    SimpleTextItem,
    IDItem,
    create_record,
)
from datamaestro_text.data.ir import AdhocAssessments
from xpm_torch.datasets import (
    LineFileDataset,
    QueryGroupedFileDataset,
    InfiniteDataset,
    ShardedIterableDataset,
)

from xpm_torch.base import Sampler
from xpmir.letor.samplers.hydrators import SampleHydrator
from xpmir.rankers import ScoredDocument


class PairwiseDistillationSample(NamedTuple):
    query: TopicRecord
    """The query"""

    documents: Tuple[DocumentRecord, DocumentRecord]
    """Positive/negative document with teacher scores"""


class PairwiseDistillationSamples(Config, Iterable[PairwiseDistillationSample]):
    """Pairwise distillation file"""

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        raise NotImplementedError()

    def get_collate_fn(self, base_collate):
        """Returns a collate function. Subclasses can override to wrap with
        hydration. Default returns base_collate unchanged."""
        return base_collate


class PairwiseHydrator(PairwiseDistillationSamples, SampleHydrator):
    """Hydrate ID-based samples with document and/or query content"""

    samples: Param[PairwiseDistillationSamples]
    """The distillation samples without texts for query and documents"""

    def transform(self, sample: PairwiseDistillationSample):
        topic, documents = sample.query, sample.documents

        if transformed := self.transform_topics([topic]):
            topic = transformed[0]

        if transformed := self.transform_documents(documents):
            documents = tuple(
                ScoredDocument(d, sd[ScoredItem].score)
                for d, sd in zip(transformed, sample.documents)
            )

        return PairwiseDistillationSample(topic, documents)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns the inner dataset (ID-only records).

        Hydration is handled at collate time via get_collate_fn().
        """
        return self.samples.as_dataset()

    def get_collate_fn(self, base_collate):
        """Returns a HydratingCollate wrapping base_collate with this adapter's stores."""
        from xpm_torch.collate import HydratingCollate

        return HydratingCollate(base_collate, self)


class PairwiseDistillationSamplesTSV(PairwiseDistillationSamples, File):
    """A TSV file (Score 1, Score 2, Query, Document 1, Document 2)"""

    with_docid: Meta[bool]
    with_queryid: Meta[bool]

    def _parse_line(self, line: str) -> PairwiseDistillationSample:
        """Parse a single TSV line into a PairwiseDistillationSample."""
        import csv
        import io

        reader = csv.reader(io.StringIO(line), delimiter="\t")
        row = next(reader)

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
                DocumentRecord(SimpleTextItem(row[3]), ScoredItem(float(row[0]))),
                DocumentRecord(SimpleTextItem(row[4]), ScoredItem(float(row[1]))),
            )

        return PairwiseDistillationSample(query, documents)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns a LineFileDataset that yields PairwiseDistillationSample."""
        return InfiniteDataset(LineFileDataset(self.path, self._parse_line))


class DistillationPairwiseSampler(Sampler):
    """Just loops over samples"""

    samples: Param[PairwiseDistillationSamples]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns the underlying dataset for use with StatefulDataLoader."""
        return self.samples.as_dataset()

    def get_collate_fn(self, base_collate):
        """Returns the collate function, with hydration if samples support it."""
        return self.samples.get_collate_fn(base_collate)


#######
# LISTWISE Distillation datasets samplers
######


class ListwiseDistillationSample(NamedTuple):
    query: TopicRecord
    """The query"""

    documents: List[DocumentRecord]
    """List of documents with their ranking position"""


class ListwiseDistillationSamples(Config, Iterable[ListwiseDistillationSample]):
    """Listwise distillation file"""

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        raise NotImplementedError()

    def get_collate_fn(self, base_collate):
        """Returns a collate function. Subclasses can override to wrap with
        hydration. Default returns base_collate unchanged."""
        return base_collate


class ListwiseHydrator(ListwiseDistillationSamples, SampleHydrator):
    """Hydrate ID-based samples with document and/or query content"""

    samples: Param[ListwiseDistillationSamples]
    """The distillation samples without texts for query and documents"""

    def transform(self, sample: ListwiseDistillationSample):
        topic, documents = sample.query, sample.documents

        if transformed := self.transform_topics([topic]):
            topic = transformed[0]

        if transformed := self.transform_documents(documents):
            documents = list(
                ScoredDocument(d, sd[ScoredItem].score)
                for d, sd in zip(transformed, sample.documents)
            )

        return ListwiseDistillationSample(topic, documents)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns the inner dataset (ID-only records).

        Hydration is handled at collate time via get_collate_fn().
        """
        return self.samples.as_dataset()

    def get_collate_fn(self, base_collate):
        """Returns a HydratingCollate wrapping base_collate with this adapter's stores."""
        from xpm_torch.collate import HydratingCollate

        return HydratingCollate(base_collate, self)


class ListwiseDistillationSamplesTSV(ListwiseDistillationSamples, File):
    """A TSV file ("query_id", "q0", "doc_id", "rank", "score", "system")"""

    top_k: Meta[int]
    with_docid: Meta[bool]
    with_queryid: Meta[bool]

    @staticmethod
    def _parse_trec_line(line: str) -> tuple:
        """Parse a TREC-format line, return (query_key, row_fields)."""
        parts = line.split("\t") if "\t" in line else line.split()
        return parts[0], parts

    def _build_group(self, query_key: str, rows: list) -> ListwiseDistillationSample:
        """Build a ListwiseDistillationSample from grouped TREC lines."""
        if self.with_queryid:
            query_record = create_record(id=query_key)
        else:
            query_record = create_record(text=query_key)

        documents = []
        for row in rows:
            if self.with_docid:
                doc = DocumentRecord(IDItem(row[2]), ScoredItem(float(row[4])))
            else:
                doc = DocumentRecord(SimpleTextItem(row[2]), ScoredItem(float(row[4])))
            documents.append(doc)

        return ListwiseDistillationSample(query_record, documents)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns a QueryGroupedFileDataset that yields ListwiseDistillationSample."""
        return InfiniteDataset(
            QueryGroupedFileDataset(
                self.path,
                self._parse_trec_line,
                self._build_group,
                top_k=self.top_k,
            )
        )


class ListwiseDistillationSamplesTSVWithAnnotations(ListwiseDistillationSamplesTSV):
    qrels: Param[AdhocAssessments]
    sampling_k: Param[int] = 8

    def __post_init__(self):
        self.qrels_dict = {}
        logging.info("Loading qrels into memory...")
        for qrel in self.qrels.iter():
            self.qrels_dict[qrel.topic_id] = [
                assess.doc_id for assess in qrel.assessments if assess.rel > 0
            ]


class DistillationListwiseSampler(Sampler):
    """Just loops over samples"""

    samples: Param[ListwiseDistillationSamples]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns the underlying dataset for use with StatefulDataLoader."""
        return self.samples.as_dataset()

    def get_collate_fn(self, base_collate):
        """Returns the collate function, with hydration if samples support it."""
        return self.samples.get_collate_fn(base_collate)


class DistillationNegativesSampler(DistillationListwiseSampler):
    """An in-batch negative sampler constructed from a listwise one"""

    sampling_k: Param[int] = 8

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)
