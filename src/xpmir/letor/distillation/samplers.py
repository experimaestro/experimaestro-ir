import logging
from dataclasses import dataclass
from typing import (
    Generic,
    Iterable,
    Iterator,
    Tuple,
    List,
    TypeVar,
)
import numpy as np

from experimaestro import Config, Meta, Param
from datamaestro.data import File
from datamaestro_text.data.ir.base import (
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

DocT = TypeVar("DocT")
DocT2 = TypeVar("DocT2")
QueryT = TypeVar("QueryT")
QueryT2 = TypeVar("QueryT2")


@dataclass
class PairwiseDistillationSample(Generic[DocT, QueryT]):
    query: QueryT
    """The query"""

    documents: Tuple[DocT, DocT]
    """Positive/negative document with teacher scores"""

    def get_queries(self) -> List[QueryT]:
        return [self.query]

    def with_queries(
        self, qs: "List[QueryT2]"
    ) -> "PairwiseDistillationSample[DocT, QueryT2]":
        return PairwiseDistillationSample(qs[0], self.documents)

    def get_documents(self) -> List[DocT]:
        return list(self.documents)

    def with_documents(
        self, ds: "List[DocT2]"
    ) -> "PairwiseDistillationSample[DocT2, QueryT]":
        return PairwiseDistillationSample(self.query, tuple(ds))


class PairwiseDistillationSamples(Config, Iterable[PairwiseDistillationSample]):
    """Pairwise distillation file"""

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        raise NotImplementedError()


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


#######
# LISTWISE Distillation datasets samplers
######


@dataclass
class ListwiseDistillationSample(Generic[DocT, QueryT]):
    query: QueryT
    """The query"""

    documents: List[DocT]
    """List of documents with their ranking position"""

    def get_queries(self) -> List[QueryT]:
        return [self.query]

    def with_queries(
        self, qs: "List[QueryT2]"
    ) -> "ListwiseDistillationSample[DocT, QueryT2]":
        return ListwiseDistillationSample(qs[0], self.documents)

    def get_documents(self) -> List[DocT]:
        return self.documents

    def with_documents(
        self, ds: "List[DocT2]"
    ) -> "ListwiseDistillationSample[DocT2, QueryT]":
        return ListwiseDistillationSample(self.query, list(ds))


class ListwiseDistillationSamples(Config, Iterable[ListwiseDistillationSample]):
    """Listwise distillation file"""

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        raise NotImplementedError()


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


class DistillationNegativesSampler(DistillationListwiseSampler):
    """An in-batch negative sampler constructed from a listwise one"""

    sampling_k: Param[int] = 8

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)
