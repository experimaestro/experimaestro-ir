from pathlib import Path
from typing import (
    Iterator,
    List,
    Optional,
    Tuple,
)
import numpy as np
from datamaestro_text.data.ir import (
    Adhoc,
    TrainingTriplets,
    PairwiseSampleDataset,
    PairwiseSample,
)
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
from xpmir.rankers import Retriever
from xpmir.utils.utils import EasyLogger, easylog
from xpmir.index import Index
from xpmir.utils.iter import (
    RandomSerializableIterator,
    SerializableIterator,
    SerializableIteratorAdapter,
    SkippingIterator,
)

logger = easylog()


# --- Base classes for samplers


class Sampler(Config, EasyLogger):
    """Abstract data sampler"""

    def initialize(self, random: Optional[np.random.RandomState]):
        self.random = random or np.random.RandomState(random.randint(0, 2**31))


class PointwiseSampler(Sampler):
    def pointwise_iter(self) -> SerializableIterator[PointwiseRecord]:
        """Iterable over pointwise records"""
        raise NotImplementedError(f"{self.__class__} should implement PointwiseRecord")


class PairwiseSampler(Sampler):
    """Abstract class for pairwise samplers which output a set of (query,
    positive, negative) triples"""

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord]:
        """Iterate over batches of size (# of queries) batch_size

        Args:
            batch_size: Number of queries per batch
        """
        raise NotImplementedError(f"{self.__class__} should implement __iter__")


class BatchwiseSampler(Sampler):
    """Batchwise samplers provide for each question a set of documents"""

    def batchwise_iter(self, batch_size: int) -> SerializableIterator[BatchwiseRecords]:
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
        """Iterates over topics, returning retrieved positives and negatives
        documents"""
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
                                f"{query.text if not positives else ''}"
                                f"\t{docno}\t0.\t{rel}\n"
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
        text = None
        while text is None:
            docid, rel, score = samples[self.random.randint(0, len(samples))]
            text = self.index.document_text(docid)
            if text is None:
                logger.warning(f"Document {docid} has no content")
        return Document(docid, text, score)

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord]:
        def iter(random):
            while True:
                title, positives, negatives = self.topics[
                    random.randint(0, len(self.topics))
                ]
                yield PairwiseRecord(
                    Query(None, title), self.sample(positives), self.sample(negatives)
                )

        return RandomSerializableIterator(self.random, iter)


class PairwiseInBatchNegativesSampler(BatchwiseSampler):
    """An in-batch negative sampler constructured from a pairwise one"""

    sampler: Param[PairwiseSampler]
    """The base pairwise sampler"""

    def initialize(self, random):
        super().initialize(random)
        self.sampler.initialize(random)

    def batchwise_iter(self, batch_size: int) -> SerializableIterator[BatchwiseRecords]:
        def iter(pair_iter):
            # Pre-compute relevance matrix (query x document)
            relevances = torch.cat(
                (torch.eye(batch_size), torch.zeros(batch_size, batch_size)), 1
            )

            while True:
                batch = ProductRecords()
                positives = []
                negatives = []
                for _, record in zip(range(batch_size), pair_iter):
                    batch.addQueries(record.query)
                    positives.append(record.positive)
                    negatives.append(record.negative)
                batch.addDocuments(*positives)
                batch.addDocuments(*negatives)
                batch.setRelevances(relevances)
                yield batch

        return SerializableIteratorAdapter(self.sampler.pairwise_iter(), iter)


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

        iterator = (
            PairwiseRecord(Query(None, query), getdoc(pos), getdoc(neg))
            for query, pos, neg in source.iter()
        )

        return SkippingIterator(iterator)


class PairwiseDatasetTripletBasedSampler(PairwiseSampler):
    """Sampler based on a dataset where each query is associated
    with (1) a set of relevant documents (2) negative documents,
    where each negative is sampled with a specific algorithm
    """

    dataset: Param[PairwiseSampleDataset]

    def pairwise_iter(self) -> SkippingIterator[PairwiseRecord]:
        class _Iterator(SkippingIterator[PairwiseRecord]):
            def __init__(
                self, random: np.random.RandomState, iterator: Iterator[PairwiseSample]
            ):
                super().__init__(iterator)
                self.random = random

            def load_state_dict(self, state):
                super().load_state_dict(state)
                self.random.set_state(state["random"])

            def state_dict(self):
                return {"random": self.random.get_state(), **super().state_dict()}

            def restore_state(self, state):
                self.random.set_state(state["random"])
                self.iter = super().restore_state(state)

            def next(self):
                sample = super().next()  # type: PairwiseSample

                pos = sample.positives[self.random.randint(len(sample.positives))]

                all_negatives = sample.negatives().values()
                negatives = all_negatives[self.random.randint(len(all_negatives))]
                neg = negatives[self.random.randint(len(negatives))]

                return PairwiseRecord(sample.query, pos, neg)

        return SkippingIterator(_Iterator(self.random, self.dataset.iter()))
