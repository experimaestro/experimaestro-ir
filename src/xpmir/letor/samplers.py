from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Any
import numpy as np
from datamaestro_text.data.ir import (
    Adhoc,
    TrainingTriplets,
    PairwiseSampleDataset,
    PairwiseSample,
    DocumentStore,
)
from datamaestro_text.data.ir.base import (
    IDDocument,
    IDTopic,
    TextTopic,
)
from experimaestro import Param, tqdm, Task, Annotated, pathgenerator
from experimaestro.annotations import cache
import torch
from xpmir.rankers import ScoredDocument
from xpmir.datasets.adapters import TextStore
from xpmir.letor.records import (
    BatchwiseRecords,
    PairwiseRecords,
    ProductRecords,
    PairwiseRecord,
    PointwiseRecord,
    TopicRecord,
    DocumentRecord,
    ScoredDocumentRecord,
)
from xpmir.rankers import Retriever, Scorer
from xpmir.learning import Sampler
from xpmir.utils.utils import easylog
from xpmir.utils.iter import (
    RandomSerializableIterator,
    SerializableIterator,
    SerializableIteratorAdapter,
    SkippingIterator,
)
from datamaestro_text.interfaces.plaintext import read_tsv

logger = easylog()


# --- Base classes for samplers


class PointwiseSampler(Sampler):
    def pointwise_iter(self) -> SerializableIterator[PointwiseRecord, Any]:
        """Iterable over pointwise records"""
        raise NotImplementedError(f"{self.__class__} should implement PointwiseRecord")


class PairwiseSampler(Sampler):
    """Abstract class for pairwise samplers which output a set of (query,
    positive, negative) triples"""

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord, Any]:
        """Iterate over batches of size (# of queries) batch_size

        Args:
            batch_size: Number of queries per batch
        """
        raise NotImplementedError(f"{self.__class__} should implement __iter__")

    def pairwise_batch_iter(self, size) -> SerializableIterator[PairwiseRecords, Any]:
        """Batchwise iterator

        Can be subclassed by some classes to be more efficient"""

        class BatchIterator:
            def __init__(self, sampler: PairwiseSampler):
                self.iter = sampler.pairwise_iter()

            def state_dict(self):
                return self.iter.state_dict()

            def load_state_dict(self, state):
                self.iter.load_state_dict(state)

            def __next__(self):
                batch = PairwiseRecords()
                for _, record in zip(range(size), self.iter):
                    batch.add(record)
                return batch

        return BatchIterator(self)


class BatchwiseSampler(Sampler):
    """Base class for batchwise samplers, that provide for each question a list
    of documents"""

    def batchwise_iter(
        self, batch_size: int
    ) -> SerializableIterator[BatchwiseRecords, Any]:
        """Iterate over batches of size (# of queries) batch_size

        Args:
            batch_size: Number of queries per batch
        """
        raise NotImplementedError(f"{self.__class__} should implement __iter__")


# --- Real instances


class ModelBasedSampler(Sampler):
    """Base class for retriever-based sampler"""

    dataset: Param[Adhoc]
    """The IR adhoc dataset"""

    retriever: Param[Retriever]
    """A retriever to sample negative documents"""

    _store: DocumentStore

    def __validate__(self) -> None:
        super().__validate__()

        assert self.retriever.get_store() is not None or isinstance(
            self.dataset.documents, DocumentStore
        ), "The retriever has no associated document store (to get document text)"

    def initialize(self, random):
        super().initialize(random)
        self._store = self.retriever.get_store() or self.dataset.documents
        assert self._store is not None, "No document store found"

    def document(self, doc_id):
        """Returns the document textual content"""
        return self._store.document_ext(doc_id)

    def document_text(self, doc_id):
        return self.document(doc_id).get_text()

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
                assessments: Dict[str, Dict[str, float]] = {}
                for qrels in self.dataset.assessments.iter():
                    doc2rel = {}
                    assessments[qrels.topic_id] = doc2rel
                    for qrel in qrels.assessments:
                        doc2rel[qrel.doc_id] = qrel.rel
                self.logger.info("Read assessments for %d topics", len(assessments))

                self.logger.info("Retrieving documents for each topic")
                queries = []
                for query in self.dataset.topics.iter():
                    queries.append(query)

                # Retrieve documents
                skipped = 0
                for query in tqdm(queries):
                    qassessments = assessments.get(query.get_id(), None)
                    if not qassessments:
                        skipped += 1
                        self.logger.warning(
                            "Skipping topic %s (no assessments)", query.get_id()
                        )
                        continue

                    # Write all the positive documents
                    positives = []
                    for docno, rel in qassessments.items():
                        if rel > 0:
                            fp.write(
                                f"{query.get_text() if not positives else ''}"
                                f"\t{docno}\t0.\t{rel}\n"
                            )
                            positives.append((docno, rel, 0))

                    if not positives:
                        self.logger.warning(
                            "Skipping topic %s (no relevant documents)", query.get_id()
                        )
                        skipped += 1
                        continue

                    scoreddocuments: List[ScoredDocument] = self.retriever.retrieve(
                        query.get_text()
                    )

                    negatives = []
                    for rank, sd in enumerate(scoreddocuments):
                        # Get the assessment (assumes not relevant)
                        rel = qassessments.get(sd.document.get_id(), 0)
                        if rel > 0:
                            continue

                        negatives.append((sd.document.get_id(), rel, sd.score))
                        fp.write(f"\t{sd.document.get_id()}\t{sd.score}\t{rel}\n")

                    if not negatives:
                        self.logger.warning(
                            "Skipping topic %s (no negatives documents)", query.get_id()
                        )
                        skipped += 1
                        continue

                    assert len(positives) > 0 and len(negatives) > 0
                    yield query.get_text(), positives, negatives

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
        self.pos_records, self.neg_records = self.readrecords()
        self.logger.info(
            "Loaded %d/%d pos/neg records", len(self.pos_records), len(self.neg_records)
        )

    def prepare(self, sample: Tuple[str, int, float]):
        assert self.document_text(sample[1]) is not None
        document = self.document_text(sample[1])

        return PointwiseRecord(
            topic=TopicRecord(TextTopic(sample[0])),
            document=DocumentRecord(document=document),
            relevance=sample[3],
        )

    def readrecords(self, runpath=None):
        pos_records, neg_records = [], []
        for title, positives, negatives in self._itertopics():
            for docno, rel, score in positives:
                pos_records.append((title, docno, score, rel))
            for docno, rel, score in negatives:
                neg_records.append((title, docno, score, rel))

        return pos_records, neg_records

    def record_iter(self) -> Iterator[PointwiseRecord]:
        npos = len(self.pos_records)
        nneg = len(self.neg_records)
        while True:
            if self.random.random() < self.relevant_ratio:
                yield self.prepare(self.pos_records[self.random.randint(0, npos)])
            else:
                yield self.prepare(self.neg_records[self.random.randint(0, nneg)])

    def pointwise_iter(self) -> SerializableIterator[PointwiseRecord, Any]:
        npos = len(self.pos_records)
        nneg = len(self.neg_records)

        def iter(random):
            while True:
                if self.random.random() < self.relevant_ratio:
                    yield self.prepare(self.pos_records[self.random.randint(0, npos)])
                else:
                    yield self.prepare(self.neg_records[self.random.randint(0, nneg)])

        return RandomSerializableIterator(self.random, iter)


class PairwiseModelBasedSampler(PairwiseSampler, ModelBasedSampler):
    """A pairwise sampler based on a retrieval model"""

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

        self.retriever.initialize()
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
            document = self.document(docid)
            text = document.get_text()
        return ScoredDocumentRecord(document, score)

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord, Any]:
        def iter(random):
            while True:
                title, positives, negatives = self.topics[
                    random.randint(0, len(self.topics))
                ]
                yield PairwiseRecord(
                    TopicRecord(TextTopic(title)),
                    self.sample(positives),
                    self.sample(negatives),
                )

        return RandomSerializableIterator(self.random, iter)


class PairwiseInBatchNegativesSampler(BatchwiseSampler):
    """An in-batch negative sampler constructured from a pairwise one"""

    sampler: Param[PairwiseSampler]
    """The base pairwise sampler"""

    def initialize(self, random):
        super().initialize(random)
        self.sampler.initialize(random)

    def batchwise_iter(
        self, batch_size: int
    ) -> SerializableIterator[BatchwiseRecords, Any]:
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
                    batch.add_topics(record.query)
                    positives.append(record.positive)
                    negatives.append(record.negative)
                batch.add_documents(*positives)
                batch.add_documents(*negatives)
                batch.set_relevances(relevances)
                yield batch

        return SerializableIteratorAdapter(self.sampler.pairwise_iter(), iter)


def always_none(*args, **kwargs):
    """Just returns None to whatever"""
    return None


class TripletBasedSampler(PairwiseSampler):
    """Sampler based on a triplet source"""

    source: Param[TrainingTriplets]
    """Triplets"""

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord, Any]:
        iterator = (
            PairwiseRecord(TopicRecord(topic), DocumentRecord(pos), DocumentRecord(neg))
            for topic, pos, neg in self.source.iter()
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


# --- Dataloader

# A class for loading the data, need to move the other places.
class PairwiseSampleDatasetFromTSV(PairwiseSampleDataset):
    """Read the pairwise sample dataset from a csv file"""

    hard_negative_samples_path: Param[Path]
    """The path which stores the existing ids"""

    def iter(self) -> Iterator[PairwiseSample]:
        """return a iterator over a set of pairwise_samples"""
        for triplet in read_tsv(self.hard_negative_samples_path):
            query = triplet[0]
            positives = triplet[2].split(" ")
            negatives = triplet[4].split(" ")
            # at the moment, I don't have some good idea to store the algo
            yield PairwiseSample(query, positives, negatives)


# A class for loading the data, need to move the other places.
class PairwiseSamplerFromTSV(PairwiseSampler):

    pairwise_samples_path: Param[Path]
    """The path which stores the existing triplets"""

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord, Any]:
        def iter() -> Iterator[PairwiseSample]:
            for triplet in read_tsv(self.pairwise_samples_path):
                q_id, pos_id, pos_score, neg_id, neg_score = triplet
                yield PairwiseRecord(
                    TopicRecord(IDTopic(q_id)),
                    ScoredDocumentRecord(IDDocument(pos_id), pos_score),
                    ScoredDocumentRecord(IDDocument(neg_id), neg_score),
                )

        return SkippingIterator(iter)


# --- Tasks for hard negatives


class ModelBasedHardNegativeSampler(Task, Sampler):
    """Retriever-based hard negative sampler"""

    dataset: Param[Adhoc]
    """The dataset which contains the topics and assessments"""

    retriever: Param[Retriever]
    """The retriever to score of the document wrt the query"""

    hard_negative_samples: Annotated[Path, pathgenerator("hard_negatives.tsv")]
    """Path to store the generated hard negatives"""

    def task_outputs(self, dep) -> PairwiseSampleDataset:
        """return a iterator of PairwiseSample"""
        return dep(
            PairwiseSampleDatasetFromTSV(
                ids=self.dataset.id,
                hard_negative_samples_path=self.hard_negative_samples,
            )
        )

    def execute(self):
        """Retrieve over the dataset and select the positive and negative
        according to the relevance score and their rank
        """
        self.logger.info("Reading topics and retrieving documents")

        # create the file
        self.hard_negative_samples.parent.mkdir(parents=True, exist_ok=True)

        # Read the assessments
        self.logger.info("Reading assessments")
        assessments = {}  # type: Dict[str, Dict[str, float]]
        for qrels in self.dataset.assessments.iter():
            doc2rel = {}
            assessments[qrels.qid] = doc2rel
            for qrel in qrels.assessments:
                doc2rel[qrel.docid] = qrel.rel
        self.logger.info("Assessment loaded")
        self.logger.info("Read assessments for %d topics", len(assessments))

        self.logger.info("Retrieving documents for each topic")
        queries = []
        for query in self.dataset.topics.iter():
            queries.append(query)

        with self.hard_negative_samples.open("wt") as fp:
            # Retrieve documents

            # count the number of queries been skipped because of no assessments
            # available
            skipped = 0
            for query in tqdm(queries):
                qassessments = assessments.get(query.qid, None)
                if not qassessments:
                    skipped += 1
                    self.logger.warning("Skipping topic %s (no assessments)", query.qid)
                    continue

                # Write all the positive documents
                positives = []
                negatives = []
                scoreddocuments: List[ScoredDocument] = self.retriever.retrieve(
                    query.get_text()
                )

                for rank, sd in enumerate(scoreddocuments):
                    if qassessments.get(sd.docid, 0) > 0:
                        # It is a positive document:
                        positives.append(sd.docid)
                    else:
                        # It is a negative document or
                        # don't exist in assessment
                        negatives.append(sd.docid)

                if not positives:
                    self.logger.debug(
                        "Skipping topic %s (no relevant documents)", query.qid
                    )
                    skipped += 1
                    continue
                if not negatives:
                    self.logger.debug(
                        "Skipping topic %s (no negative documents)", query.qid
                    )
                    skipped += 1
                    continue

                # Write the result to the file
                positive_str = " ".join(positives)
                negative_str = " ".join(negatives)
                fp.write(
                    f"{qrels.qid}\tpositives:\t{positive_str}\t"
                    f"negatives:\t{negative_str}"
                )

        self.logger.info("Processed %d topics (%d skipped)", len(queries), skipped)


class TeacherModelBasedHardNegativesTripletSampler(Task, Sampler):
    """For a given set of triplet, assign the score
    for the documents according to the teacher model"""

    sampler: Param[PairwiseSampler]
    """The list of exsting hard negatives which we can sample from"""

    document_store: Param[DocumentStore]
    """The document store"""

    topic_store: Param[TextStore]
    """The query_document store"""

    teacher_model: Param[Scorer]
    """The teacher model which scores the positive and negative document"""

    hard_negative_triplet: Annotated[Path, pathgenerator("triplet.tsv")]
    """The path to store the generated triplets"""

    batch_size: int
    """How many pairs of documents are been calculate in a batch"""

    def task_outputs(self, dep) -> PairwiseSampler:
        return dep(
            PairwiseSamplerFromTSV(pairwise_samples_path=self.hard_negative_triplet)
        )

    def iter_pairs_with_text(self) -> Iterator[PairwiseRecord]:
        """Add the information of the text back to the records"""
        for record in self.sampler.pairwise_iter():
            record.query.text = self.topic_store[record.query.id]
            record.positive.text = self.document_store.document_text(
                record.positive.docid
            )
            record.negative.text = self.document_store.document_text(
                record.negative.docid
            )
            yield record

    def iter_batches(self) -> Iterator[PairwiseRecords]:
        """Return the batch which contains the records"""
        while True:
            batch = PairwiseRecords()
            for _, record in zip(range(self.batch_size), self.iter_pairs_with_text()):
                batch.add(record)
            yield batch

    def execute(self):
        """Pre-calculate the score for the teacher model, and store them"""

        self.logger.info("Calculating the score for the teacher model")
        # create the file
        self.hard_negative_triplet.parent.mkdir(parents=True, exist_ok=True)

        # make the tqdm progressing wrt one record, not a batch of records
        with self.hard_negative_triplet.open("wt") as fp:
            for batch in tqdm(self.iter_batches()):

                # scores in shape: [batch_size, 2]
                self.teacher_model.eval()
                scores = self.teacher_model(batch)
                scores = scores.reshape(2, -1).T

                # write in the file
                for i, record in enumerate(batch):
                    fp.write(
                        f"{record.query.id}\t{record.positive.id}\t{scores[i,0]}"
                        f"\t{record.negative.id}\t{scores[i,1]}"
                    )
        self.logger.info("Teacher models score generating finish")
