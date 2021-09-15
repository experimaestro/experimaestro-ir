import itertools
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple
import numpy as np
from datamaestro_text.data.ir import Adhoc, AdhocTopic, TrainingTriplets
from experimaestro import Config, Param, tqdm
from experimaestro.annotations import cache
import torch
from xpmir.letor.records import (
    BatchwiseRecords,
    CartesianProductRecords,
    Document,
    PairwiseRecord,
    PointwiseRecord,
    Query,
)
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.test.neural.test_forward import pairwise
from xpmir.utils import EasyLogger
from xpmir.index import Index


class Sampler(Config, EasyLogger):
    """"Abtract data sampler"""

    def initialize(self, random: np.random.RandomState):
        self.random = random

    def record_iter(self) -> Iterator[PointwiseRecord]:
        """Returns an iterator over records (query, document, relevance)"""
        raise NotImplementedError(f"{self.__class__} does not implement record_iter()")

    def pairwiserecord_iter(self) -> Iterator[PairwiseRecord]:
        """Returns an iterator over records (query, document, relevance)"""
        raise NotImplementedError(
            f"{self.__class__} does not implement pairwiserecord_iter()"
        )


class BatchwiseSampler(Sampler, Iterable[BatchwiseRecords]):
    def __iter__(self, batch_size: int) -> Iterator[BatchwiseRecords]:
        """Iterate over batches of size (# of queries) batch_size

        Args:
            batch_size: Number of queries per batch
        """
        raise NotImplementedError(f"{self.__class__} should implement __iter__")


class ModelBasedSampler(Sampler):
    """Retriever-based sampler

    Attributes:
        relevant_ratio: The sampling ratio of relevant to non relevant
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
            with runpath.open("wt") as fp:

                # Read the assessments
                self.logger.info("Reading assessments")
                assessments = {}  # type: Dict[str, Dict[str, float]]
                for qrels in self.dataset.assessments.iter():
                    doc2rel = {}
                    assessments[qrels.qid] = doc2rel
                    for docid, rel in qrels.assessments:
                        doc2rel[docid] = rel
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
                        self.logger.warning(
                            "Skipping topic %s (no assessments)", query.qid
                        )
                        continue

                    totalrel = sum(rel for docno, rel in qassessments.items())
                    if totalrel == 0:
                        self.logger.debug(
                            "Skipping topic %s (no relevant documents)", query.qid
                        )
                        skipped += 1
                        continue

                    scoreddocuments = self.retriever.retrieve(
                        query.text
                    )  # type: List[ScoredDocument]

                    positives = []
                    negatives = []
                    for rank, sd in enumerate(scoreddocuments):
                        # Get the assessment (assumes not relevant)
                        rel = qassessments.get(sd.docid, 0)
                        (positives if rel > 0 else negatives).append(
                            (sd.docid, rel, sd.score)
                        )
                        fp.write(
                            f"{query.text if rank == 0 else ''}\t{sd.docid}\t{sd.score}\t{rel}\n"
                        )

                    yield query.text, positives, negatives

                # Finally...
                self.logger.info(
                    "Processed %d topics (%d skipped)", len(queries), skipped
                )
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


class PointwiseModelBasedSampler(ModelBasedSampler):
    relevant_ratio: Param[float] = 0.5

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


class PairwiseModelBasedSampler(ModelBasedSampler):
    """A pairwise sampler based on a retrieval model"""

    def initialize(self, random):
        super().initialize(random)

        self.retriever.initialize()
        self.index = self.retriever.index
        self.topics = self._readrecords()

    def _readrecords(self):
        topics = []
        for title, positives, negatives in self._itertopics():
            topics.append((title, positives, negatives))
        return topics

    def sample(self, samples: List[Tuple[str, int, float]]):
        docid, rel, score = samples[self.random.randint(0, len(samples))]
        text = self.index.document_text(docid)
        return Document(docid, text, score)

    def pairwiserecord_iter(self) -> Iterator[PairwiseRecord]:
        while True:
            title, positives, negatives = self.topics[
                self.random.randint(0, len(self.topics))
            ]
            yield PairwiseRecord(title, self.sample(positives), self.sample(negatives))


class PairwiseInBatchNegativesSampler(BatchwiseSampler):
    """An in-batch negative sampler constructured from a pairwise one

    Args:
        BatchwiseSampler: A pairwise sampler
    """

    sampler: Param[PairwiseModelBasedSampler]

    def initialize(self, random):
        super().initialize(random)
        self.sampler.initialize(random)

    def __iter__(self, batch_size: int) -> Iterator[BatchwiseRecords]:
        it = self.sampler.pairwiserecord_iter()

        # Pre-compute relevance matrix
        relevances = torch.diag(torch.ones(batch_size * 2, dtype=torch.float))

        while True:
            batch = CartesianProductRecords()
            for _, record in zip(range(batch_size), it):
                batch.addQueries(record.query)
                batch.addDocuments(record.positive, record.negative)
            batch.setRelevances(relevances)
            yield batch


class TripletBasedSampler(Sampler):
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
        return Document(docid, self.index.document_text(docid), None)

    @staticmethod
    def _fromtext(text: str):
        return Document(None, text, None)

    def pairwiserecord_iter(self) -> Iterator[PairwiseRecord]:
        # Nature of the documents
        getdoc = self._fromid if self.source.ids else self._fromtext

        while True:
            for query, pos, neg in self.source.iter():
                yield PairwiseRecord(Query(query), getdoc(pos), getdoc(neg))
