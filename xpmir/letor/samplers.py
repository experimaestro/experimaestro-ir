from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, TypeVar, Protocol
import numpy as np
from datamaestro_text.data.ir import Adhoc, TrainingTriplets
from experimaestro import Config, Param, tqdm
from experimaestro.annotations import cache
import torch
from xpmir.letor.records import (
    BatchwiseRecords,
    ProductRecords,
    Document,
    PairwiseRecord,
    PointwiseRecord,
    Query,
)
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.utils import EasyLogger, easylog
from xpmir.index import Index

logger = easylog()

# --- Utility classes

T = TypeVar("T", covariant=True)


class SerializableIterator(Iterator[T], Protocol):
    def state_dict(self) -> Dict:
        ...

    def load_state_dict(self, state):
        ...


class RandomSerializableIterator(Iterator[T]):
    def __init__(self, iter: Iterator[T], state: Optional[Dict]):
        self.iter = iter

    def state_dict(self):
        return {}

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)


class SkippingIterator(Iterable[T]):
    """An iterator that skips the first entries and can output its state"""

    def __init__(self, iterable: Iterable[T]):
        self.position = 0
        self.iterable = iterable
        self.iter = iter(iterable)
        self.count = 0

    def state_dict(self):
        return {"count": self.position}

    def load_state_dict(self, state):
        self.iter = None
        self.position = state.get("count", 0)

    def __iter__(self):
        return self

    def __next__(self):
        self.iter = self.iter or iter(self.iterable)

        # Nature of the documents
        if self.position < self.count:
            # Skip self.count items
            logger.info("Skipping %d records to match state (sampler)", self.count)
            for _ in range(self.position - self.count):
                next(self.iter)
            self.position = self.count

        # And now go ahead
        element = next(self.iter)
        self.position += 1
        return element


# --- Base classes for samplers


class Sampler(Config, EasyLogger):
    """"Abtract data sampler"""

    def initialize(self, random: np.random.RandomState):
        self.random = np.random.RandomState(random.randint(0, 2 ** 31))

    def state_dict(self) -> Dict:
        raise NotImplementedError(f"state_dict() not implemented in {self.__class__}")

    def load_dict(self, Dict):
        raise NotImplementedError(f"load_dict() not implemented in {self.__class__}")


class PointwiseSampler(Sampler):
    def pointwise_iter(self) -> SerializableIterator[PointwiseRecord]:
        raise NotImplementedError(f"{self.__class__} should implement PointwiseRecord")


class PairwiseSampler(Sampler):
    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord]:
        """Iterate over batches of size (# of queries) batch_size

        Args:
            batch_size: Number of queries per batch
        """
        raise NotImplementedError(f"{self.__class__} should implement __iter__")


class BatchwiseSampler(Sampler):
    def batchwise_iter(self) -> SerializableIterator[BatchwiseRecords]:
        """Iterate over batches of size (# of queries) batch_size

        Args:
            batch_size: Number of queries per batch
        """
        raise NotImplementedError(f"{self.__class__} should implement __iter__")


# --- Real instances


class ModelBasedSampler(Sampler):
    """Retriever-based sampler

    Attributes:
        dataset: The topics and assessments
        retriever: The document retriever
    """

    dataset: Param[Adhoc]
    retriever: Param[Retriever]

    def getdocuments(self, docids: List[str]):
        self._documents = [
            self.retriever.collection().document_text(docid) for docid in docids
        ]

    @cache("run")
    def _itertopics(
        self, runpath: Path
    ) -> Iterator[
        Tuple[str, List[Tuple[str, int, float]], List[Tuple[str, int, float]]]
    ]:
        """Iterates over topics, returning retrieved positives and negatives documents"""
        self.logger.info("Reading topics and retrieving documents")

        if not runpath.is_file():
            tmprunpath = runpath.with_suffix(".tmp")

            with tmprunpath.open("wt") as fp:

                # Read the assessments
                self.logger.info("Reading assessments")
                assessments = {}  # type: Dict[str, Dict[str, float]]
                for qrels in self.dataset.assessments.iter():
                    doc2rel = {}
                    assessments[qrels.qid] = doc2rel
                    for qrel in qrels.assessments:
                        doc2rel[qrel.docno] = qrel.rel
                self.logger.info("Read assessments for %d topics", len(assessments))

                self.logger.info("Retrieving documents for each topic")
                queries = []
                for query in self.dataset.topics.iter():
                    queries.append(query)

                # Retrieve documents
                skipped = 0
                for query in tqdm(queries):
                    qassessments = assessments.get(query.qid, None)
                    if not qassessments:
                        skipped += 1
                        self.logger.warning(
                            "Skipping topic %s (no assessments)", query.qid
                        )
                        continue

                    # Write all the positive documents
                    positives = []
                    for docno, rel in qassessments.items():
                        if rel > 0:
                            fp.write(
                                f"{query.text if not positives else ''}\t{docno}\t0.\t{rel}\n"
                            )
                            positives.append((docno, rel, 0))

                    if not positives:
                        self.logger.debug(
                            "Skipping topic %s (no relevant documents)", query.qid
                        )
                        skipped += 1
                        continue

                    scoreddocuments = self.retriever.retrieve(
                        query.text
                    )  # type: List[ScoredDocument]

                    negatives = []
                    for rank, sd in enumerate(scoreddocuments):
                        # Get the assessment (assumes not relevant)
                        rel = qassessments.get(sd.docid, 0)
                        if rel > 0:
                            continue

                        negatives.append((sd.docid, rel, sd.score))
                        fp.write(f"\t{sd.docid}\t{sd.score}\t{rel}\n")

                    assert len(positives) > 0 and len(negatives) > 0
                    yield query.text, positives, negatives

                # Finally, move the cache file in place...
                self.logger.info(
                    "Processed %d topics (%d skipped)", len(queries), skipped
                )
                tmprunpath.rename(runpath)
        else:
            # Read from cache
            self.logger.info("Reading records from file %s", runpath)
            with runpath.open("rt") as fp:
                positives = []
                negatives = []
                oldtitle = ""

                for line in fp.readlines():
                    title, docno, score, rel = line.rstrip().split("\t")
                    if title:
                        if oldtitle:
                            yield oldtitle, positives, negatives
                        positives = []
                        negatives = []
                    else:
                        title = oldtitle
                    title = title or oldtitle
                    rel = int(rel)
                    (positives if rel > 0 else negatives).append(
                        (docno, rel, float(score))
                    )
                    oldtitle = title

                yield oldtitle, positives, negatives


class PointwiseModelBasedSampler(PointwiseSampler, ModelBasedSampler):
    relevant_ratio: Param[float] = 0.5
    """The target relevance ratio"""

    def initialize(self, random):
        super().initialize(random)

        self.retriever.initialize()
        self.index = self.retriever.index
        self.pos_records, self.neg_records = self.readrecords()
        self.logger.info(
            "Loaded %d/%d pos/neg records", len(self.pos_records), len(self.neg_records)
        )

    def prepare(self, record: PointwiseRecord):
        if record.document.text is None:
            record.document.text = self.index.document_text(record.document.docid)
        return record

    def readrecords(self, runpath):
        pos_records, neg_records = [], []
        for title, positives, negatives in self._itertopics():
            for docno, rel, score in positives:
                self.pos_records.append(PointwiseRecord(title, docno, None, score, rel))
            for docno, rel, score in negatives:
                self.neg_records.append(PointwiseRecord(title, docno, None, score, rel))

        return pos_records, neg_records

    def record_iter(self) -> Iterator[PointwiseRecord]:
        npos = len(self.pos_records)
        nneg = len(self.neg_records)
        while True:
            if self.random.random() < self.relevant_ratio:
                yield self.prepare(self.pos_records[self.random.randint(0, npos)])
            else:
                yield self.prepare(self.neg_records[self.random.randint(0, nneg)])


class PairwiseModelBasedSampler(PairwiseSampler, ModelBasedSampler):
    """A pairwise sampler based on a retrieval model"""

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

        self.retriever.initialize()
        self.index = self.retriever.index
        self.topics: List[Tuple[str, List, List]] = self._readrecords()

    def _readrecords(self):
        topics = []
        for title, positives, negatives in self._itertopics():
            topics.append((title, positives, negatives))
        return topics

    def sample(self, samples: List[Tuple[str, int, float]]):
        docid, rel, score = samples[self.random.randint(0, len(samples))]
        text = self.index.document_text(docid)
        return Document(docid, text, score)

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord]:
        def iter():
            while True:
                title, positives, negatives = self.topics[
                    self.random.randint(0, len(self.topics))
                ]
                yield PairwiseRecord(
                    Query(None, title), self.sample(positives), self.sample(negatives)
                )

        return RandomSerializableIterator(iter(), state)


class PairwiseInBatchNegativesSampler(BatchwiseSampler):
    """An in-batch negative sampler constructured from a pairwise one"""

    sampler: Param[PairwiseSampler]
    """The base pairwise sampler"""

    def initialize(self, random):
        super().initialize(random)
        self.sampler.initialize(random)

    def batchwise_iter(self, batch_size: int) -> SerializableIterator[BatchwiseRecords]:
        it = self.sampler.pairwise_iter()

        def iter():
            # Pre-compute relevance matrix
            relevances = torch.diag(torch.ones(batch_size * 2, dtype=torch.float))

            while True:
                batch = ProductRecords()
                for _, record in zip(range(batch_size), it):
                    batch.addQueries(record.query)
                    batch.addDocuments(record.positive, record.negative)
                batch.setRelevances(relevances)
                yield batch

        return RandomSerializableIterator(iter(), state)


class TripletBasedSampler(PairwiseSampler):
    """Sampler based on a triplet file

    Attributes:

    source: the source of the triplets
    index: the index (if the source is only)
    """

    source: Param[TrainingTriplets]
    index: Param[Optional[Index]] = None

    def __validate__(self):
        assert (
            not self.source.ids or self.index is not None
        ), "An index should be provided if source is IDs only"

    def _fromid(self, docid: str):
        assert self.index is not None
        return Document(docid, self.index.document_text(docid), None)

    @staticmethod
    def _fromtext(text: str):
        return Document(None, text, None)

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord]:
        getdoc = self._fromid if self.source.ids else self._fromtext
        source = self.source

        class _Iterable(Iterable[PairwiseRecord]):
            def __iter__(self):
                return (
                    PairwiseRecord(Query(None, query), getdoc(pos), getdoc(neg))
                    for query, pos, neg in source.iter()
                )

        return SkippingIterator(_Iterable())
