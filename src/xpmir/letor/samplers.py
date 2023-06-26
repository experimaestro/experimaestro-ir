from pathlib import Path
from typing import Iterator, List, Tuple, Dict, Optional
import numpy as np
import random
from datamaestro_text.data.ir import (
    Adhoc,
    TrainingTriplets,
    PairwiseSampleDataset,
    PairwiseSample,
    DocumentStore,
)
from datamaestro_text.data.ir.data import IDDocument, IDTopic, TextTopic
from experimaestro import Param, tqdm, Task, Annotated, pathgenerator
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
    ListwiseSerializableIterator,
)
from datamaestro_text.interfaces.plaintext import read_tsv
from xpmir.utils.utils import batchiter

logger = easylog()


# --- Base classes for samplers


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
    """Base class for retriever-based sampler

    Attributes:
        dataset: The topics and assessments
        retriever: The document retriever
    """

    dataset: Param[Adhoc]
    retriever: Param[Retriever]
    max_query: Param[int] = -1
    """The number of queries to be considered, if equals -1 means need to
    consider all"""

    require_initialization: Param[int] = True
    """Need to build the negatives as the sampler at the initialization stage"""

    _store: DocumentStore
    batch_size: Param[int] = 0
    """The batch_size for retrieving"""

    def __validate__(self) -> None:
        super().__validate__()

        assert (
            self.retriever.get_store() is not None
        ), "The retriever has no associated document store"

    def initialize(self, random):
        super().initialize(random)
        self._store = self.retriever.get_store()

    def update(self):
        """The abstract class for update the sampler"""
        raise NotImplementedError("update not implemented in f{self.__class__}")

    def document(self, doc_id):
        """Returns the document textual content"""
        return self._store.document_ext(doc_id)

    def _itertopics(
        self,
    ) -> Iterator[
        Tuple[str, List[Tuple[str, int, float]], List[Tuple[str, int, float]]]
    ]:
        """Iterates over topics, returning retrieved positives and negatives
        documents"""
        self.logger.info("Reading topics and retrieving documents")

        # Read the assessments
        self.logger.info("Reading assessments")
        assessments: Dict[str, Dict[str, float]] = {}
        for qrels in self.dataset.assessments.iter():
            doc2rel = {}
            assessments[qrels.qid] = doc2rel
            for qrel in qrels.assessments:
                doc2rel[qrel.docid] = qrel.rel
        self.logger.info("Read assessments for %d topics", len(assessments))

        self.logger.info("Retrieving documents for each topic")
        queries = []
        for query in self.dataset.topics.iter():
            queries.append(query)

        if self.max_query >= 0:
            queries = random.sample(queries, self.max_query)

        # Retrieve documents
        skipped = 0
        for query in tqdm(queries):
            qassessments = assessments.get(query.qid, None)
            if not qassessments:
                skipped += 1
                self.logger.warning("Skipping topic %s (no assessments)", query.qid)
                continue

            # Write all the positive documents
            positives = []
            for docno, rel in qassessments.items():
                if rel > 0:
                    positives.append((docno, rel, 0))

            if not positives:
                self.logger.debug(
                    "Skipping topic %s (no relevant documents)", query.qid
                )
                skipped += 1
                continue

            scoreddocuments: List[ScoredDocument] = self.retriever.retrieve(
                query.get_text()
            )

            negatives = []
            for _, sd in enumerate(scoreddocuments):
                # Get the assessment (assumes not relevant)
                rel = qassessments.get(sd.docid, 0)
                if rel > 0:
                    continue

                negatives.append((sd.docid, rel, sd.score))

            assert len(positives) > 0 and len(negatives) > 0
            yield query.get_text(), positives, negatives

        # Finally, move the cache file in place...
        self.logger.info("Processed %d topics (%d skipped)", len(queries), skipped)

    def _itertopics_batchwise(
        self,
    ) -> Iterator[
        Tuple[str, List[Tuple[str, int, float]], List[Tuple[str, int, float]]]
    ]:
        """Iterates over topics, returning retrieved positives and negatives
        documents"""

        self.logger.info("Reading topics and retrieving documents")

        # Read the assessments to memory
        self.logger.info("Reading assessments")
        assessments = {}  # type: Dict[str, Dict[str, float]]
        for qrels in self.dataset.assessments.iter():
            doc2rel = {}
            assessments[qrels.qid] = doc2rel
            for qrel in qrels.assessments:
                doc2rel[qrel.docid] = qrel.rel
        self.logger.info("Read assessments for %d topics", len(assessments))

        self.logger.info("Retrieving documents for each topic")
        queries = []
        for query in self.dataset.topics.iter():
            queries.append(query)

        if self.max_query >= 0:
            queries = random.sample(queries, self.max_query)

        # Important! Assuming all the queries has the assessments and relevant docs.
        for batch in tqdm(batchiter(self.batch_size, queries)):
            queries_dict = {query.qid: query.get_text() for query in batch}
            scoreddocuments_batch = self.retriever.retrieve_all(
                queries_dict
            )  # Dict[str, List[ScoredDocument]]
            for (qid, scoreddocuments) in scoreddocuments_batch.items():
                negatives = []
                positives = []  # maybe contains > 1 rel docs so need also a list
                qassessments = assessments.get(qid, None)
                # for the positives
                for docno, rel in qassessments.items():
                    if rel > 0:
                        positives.append((docno, rel, 0))

                # for the negatives
                for sd in scoreddocuments:
                    rel = qassessments.get(sd.docid, 0)  # check whether it is positive
                    if rel > 0:
                        continue
                    negatives.append((sd.docid, rel, sd.score))

                yield queries_dict[qid], positives, negatives


class PointwiseModelBasedSampler(PointwiseSampler, ModelBasedSampler):
    relevant_ratio: Param[float] = 0.5
    """The target relevance ratio"""

    def initialize(self, random):
        super().initialize(random)

        self.retriever.initialize()
        if self.require_initialization:
            self.pos_records, self.neg_records = self.readrecords()
        self.logger.info(
            "Loaded %d/%d pos/neg records", len(self.pos_records), len(self.neg_records)
        )

    def prepare(self, record: PointwiseRecord):
        if record.document.text is None:
            record.document.text = self.document_text(record.document.docid)
        return record

    def readrecords(self):
        pos_records, neg_records = [], []
        # if we want to do it one by one: slow, but support the case where the
        # query has no assessment
        if self.batch_size == 0:
            for title, positives, negatives in self._itertopics():
                for docno, rel, score in positives:
                    pos_records.append(PointwiseRecord(title, docno, None, score, rel))
                for docno, rel, score in negatives:
                    neg_records.append(PointwiseRecord(title, docno, None, score, rel))
        else:  # do it batch by batch
            for title, positives, negatives in self._itertopics_batchwise():
                for docno, rel, score in positives:
                    pos_records.append(PointwiseRecord(title, docno, None, score, rel))
                for docno, rel, score in negatives:
                    neg_records.append(PointwiseRecord(title, docno, None, score, rel))

        return pos_records, neg_records

    def update(self):
        self.pos_records, self.neg_records = self.readrecords()

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
        if self.require_initialization:
            self.topics: List[Tuple[str, List, List]] = self._readrecords()

    def _readrecords(self):
        topics = []
        # if we want to do it one by one: slow, but support the case where the
        # query has no assessment
        if self.batch_size == 0:
            for title, positives, negatives in self._itertopics():
                topics.append((title, positives, negatives))
        else:  # do it batch by batch
            for title, positives, negatives in self._itertopics_batchwise():
                topics.append((title, positives, negatives))
        return topics

    def sample(self, samples: List[Tuple[str, int, float]]):
        text = None
        while text is None:
            docid, _, score = samples[self.random.randint(0, len(samples))]
            text = self.document_text(docid)
        return DocumentRecord(docid, text, score)

    def update(self):
        self.topics = self._readrecords()

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord]:
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


class PairwiseListSamplers(PairwiseSampler):
    """A list of pairwise samplers which could be changed during the learning
    procedure"""

    samplers: Param[List[PairwiseSampler]]
    """The list of samplers to be used"""

    def initialize(self, random: Optional[np.random.RandomState]):
        for sampler in self.samplers:
            sampler.initialize(random)

    def pairwise_iter(self) -> ListwiseSerializableIterator[PairwiseRecord]:
        return ListwiseSerializableIterator(
            [sampler.pairwise_iter() for sampler in self.samplers]
        )


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

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord]:
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

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord]:
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
