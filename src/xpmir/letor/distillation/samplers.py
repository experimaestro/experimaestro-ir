import logging, random
import hashlib
from dataclasses import dataclass
from typing import (
    Optional,
    Generic,
    Iterable,
    Iterator,
    Tuple,
    List,
    TypeVar,
)
import numpy as np

from experimaestro import Config, Meta, Param, field
from datamaestro.data import File
from datamaestro_text.data.ir import (
    IDRecord,
    TextRecord,
    SimpleTextItem,
)
from xpmir.rankers import ScoredDocument
from datamaestro_text.data.ir import AdhocAssessments
from xpm_torch.datasets import (
    LineFileDataset,
    QueryGroupedFileDataset,
    InfiniteDataset,
    ShardedIterableDataset,
    TransformDataset,
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
            query = IDRecord(id=row[2])
        else:
            query = TextRecord(text_item=SimpleTextItem(row[2]))

        if self.with_docid:
            documents = (
                ScoredDocument(IDRecord(id=row[3]), float(row[0])),
                ScoredDocument(IDRecord(id=row[4]), float(row[1])),
            )
        else:
            documents = (
                ScoredDocument(
                    TextRecord(text_item=SimpleTextItem(row[3])), float(row[0])
                ),
                ScoredDocument(
                    TextRecord(text_item=SimpleTextItem(row[4])), float(row[1])
                ),
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
            query_record = IDRecord(id=query_key)
        else:
            query_record = TextRecord(text_item=SimpleTextItem(query_key))

        documents = []
        for row in rows:
            if self.with_docid:
                doc = ScoredDocument(IDRecord(id=row[2]), float(row[4]))
            else:
                doc = ScoredDocument(
                    TextRecord(text_item=SimpleTextItem(row[2])), float(row[4])
                )
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

    def initialize(self, random: Optional[np.random.RandomState]):
        super().initialize(random)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns the underlying dataset for use with StatefulDataLoader."""
        return self.samples.as_dataset()


class DistillationNegativesSampler(DistillationListwiseSampler):
    """Samples only `passages_per_query` documents per query, skips query if no relevant document retrieved
     - Needs the relevances judgements to ensure sampling one positive and (passages_per_query - 1) negatives per query
     - Uses ScoredDocument to store Relevance Labels, 
        - WARNING: ignores eventual scores from original Dataset.
    """
    
    samples: Param[ListwiseDistillationSamplesTSVWithAnnotations]
    passages_per_query: Param[int] = field(default=8)

    def _sample_docs(self, item):
        qrel = self.samples.qrels_dict.get(item.query["id"], set())
        negatives = []
        positives = []

        for doc in item.documents:
            if doc.document["id"] in qrel:
                positives.append(ScoredDocument(doc.document, score=1))
            else:
                negatives.append(ScoredDocument(doc.document, score=0))

        if not positives:  # this will be skipped by TransformDataset.iter_shard
            return

        # if we have positives, return one per positive doc
        sampled_negatives = [
            negatives[idx]
            for idx in self.random.choice(
                len(negatives), self.passages_per_query - 1
            )
        ]

        # return positive document fist and then
        return ListwiseDistillationSample(
            query=item.query, documents=[positives[0]] + sampled_negatives
        )

    def initialize(self, random: Optional[np.random.RandomState]):
        super().initialize(random)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns the underlying dataset for use with StatefulDataLoader."""
        return TransformDataset(self.samples.as_dataset(), transform=self._sample_docs)
