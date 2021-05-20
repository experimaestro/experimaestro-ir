import itertools
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Tuple
import torch
import numpy as np
from datamaestro_text.data.ir import Adhoc, AdhocTopic, TrainingTriplets
from experimaestro import Config, Param, config, help, param, tqdm
from experimaestro.annotations import cache
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.utils import EasyLogger
from xpmir.dm.data import Index


class Document:
    docid: str
    text: str
    score: float

    def __init__(self, docid, text, score):
        self.docid = docid
        self.text = text
        self.score = score


class PointwiseRecord:
    """A record from a pointwise sampler"""

    # The query
    query: str

    # The document
    document: Document

    # The relevance
    relevance: Optional[float]

    def __init__(self, query, docid, document, score, relevance):
        self.query = query
        self.document = Document(docid, document, score)
        self.relevance = relevance


class TokenizedTexts:
    def __init__(self, tokens: List[List[str]], ids: torch.LongTensor, lens: List[int]):
        self.tokens = tokens
        self.ids = ids
        self.lens = lens


class Records:
    """Records are the objects passed to the module forwards"""

    # The queries
    queries: List[str]

    # Text of the documents
    documents: List[Document]

    # The relevances
    relevances: List[float]

    def __init__(self):
        self.queries = []
        self.documents = []
        self.relevances = []

    def add(self, record: PointwiseRecord):
        self.queries.append(record.query)
        self.relevances.append(record.relevance)
        self.documents.append(record.document)


class PairwiseRecord:
    query: str
    positive: Document
    negative: Document

    def __init__(self, query: str, positive: Document, negative: Document):
        self.query = query
        self.positive = positive
        self.negative = negative


class PairwiseRecords:
    """"""

    # The queries
    _queries: List[str]

    # The document IDs (positive)
    positives: List[Document]

    # The scores of the retriever
    negatives: List[Document]

    def __init__(self):
        self._queries = []
        self.positives = []
        self.negatives = []

    def add(self, record: PairwiseRecord):
        self._queries.append(record.query)
        self.positives.append(record.positive)
        self.negatives.append(record.negative)

    @property
    def queries(self):
        return itertools.chain(self._queries, self._queries)

    @property
    def documents(self):
        return itertools.chain(self.positives, self.negatives)


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
                        query.title
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
                            f"{query.title if rank == 0 else ''}\t{sd.docid}\t{sd.score}\t{rel}\n"
                        )

                    yield query, positives, negatives

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


class TripletBasedSampler(Sampler):
    """Sampler based on a triplet file

    Attributes:

    source: the source of the triplets
    index: the index (if the source is only)
    """

    source: Param[TrainingTriplets]
    index: Param[Index] = None

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
                yield PairwiseRecord(query, getdoc(pos), getdoc(neg))
