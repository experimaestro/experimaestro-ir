import itertools
from typing import Any, Iterator, List, NamedTuple, Optional
import torch
import numpy as np
from datamaestro_text.data.ir import Adhoc, TrainingTriplets
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

    def __init__(self, query, positive, negative):
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
    """Sampler based on a retriever

    Args:

    relevant_ratio: The sampling ratio of relevant to non relevant
    dataset: The topics and assessments
    retriever: The document retriever
    """

    relevant_ratio: Param[float] = 0.5
    dataset: Param[Adhoc]
    retriever: Param[Retriever]

    def initialize(self, random):
        super().initialize(random)

        self.retriever.initialize()
        self.index = self.retriever.index
        self.pos_records, self.neg_records = self.readrecords()
        self.logger.info(
            "Loaded %d/%d pos/neg records", len(self.pos_records), len(self.neg_records)
        )

    def getdocuments(self, docids: List[str]):
        self._documents = [
            self.retriever.collection().document_text(docid) for docid in docids
        ]

    @cache("run")
    def readrecords(self, runpath):
        pos_records, neg_records = [], []
        if not runpath.is_file():
            self.logger.info("Reading topics and retrieving documents")
            self.logger.info("Caching in %s", runpath)

            # Read the assessments
            self.logger.info("Reading assessments")
            assessments = {}
            for qrels in self.dataset.assessments.iter():
                doc2rel = {}
                assessments[qrels.qid] = doc2rel
                for docid, rel in qrels.assessments:
                    doc2rel[docid] = rel
            self.logger.info("Read assessments for %d topics", len(assessments))

            with runpath.open("wt") as fp:
                self.logger.info("Retrieving documents for each topic")
                queries = []
                for query in self.dataset.topics.iter():
                    queries.append(query)

                skipped = 0
                for query in tqdm(queries):
                    qassessments = assessments.get(query.qid, None) or {}
                    totalrel = sum(rel for docno, rel in qassessments.items())
                    if totalrel == 0:
                        self.logger.debug(
                            "Skipping topic %s (no relevant documents)", query.qid
                        )
                        skipped += 1
                        continue
                    scoredDocuments = self.retriever.retrieve(
                        query.title
                    )  # type: List[ScoredDocument]
                    for rank, sd in enumerate(scoredDocuments):
                        # Get the assessment (assumes not relevant)
                        rel = qassessments.get(sd.docid, 0)
                        (pos_records if rel > 0 else neg_records).append(
                            PointwiseRecord(query.title, sd.docid, None, sd.score, rel)
                        )
                        fp.write(
                            f"{query.title if rank == 0 else ''}\t{sd.docid}\t{sd.score}\t{rel}\n"
                        )
                self.logger.info(
                    "Process %d topics (%d skipped)", len(queries), skipped
                )
        else:
            # Read from file
            self.logger.info("Reading records from file %s", runpath)
            with runpath.open("rt") as fp:
                oldtitle = ""
                for line in fp.readlines():
                    title, docno, score, rel = line.rstrip().split("\t")
                    title = title or oldtitle
                    rel = int(rel)

                    (pos_records if rel > 0 else neg_records).append(
                        PointwiseRecord(title, docno, None, float(score), rel)
                    )
                    oldtitle = title

        return pos_records, neg_records

    def prepare(self, record: PointwiseRecord):
        if record.document.text is None:
            record.document.text = self.index.document_text(record.document.docid)
        return record

    def record_iter(self) -> Iterator[PointwiseRecord]:
        npos = len(self.pos_records)
        nneg = len(self.neg_records)
        while True:
            if self.random.random() < self.relevant_ratio:
                yield self.prepare(self.pos_records[self.random.randint(0, npos)])
            else:
                yield self.prepare(self.neg_records[self.random.randint(0, nneg)])

    def pairwiserecord_iter(self) -> Iterator[PairwiseRecord]:
        raise NotImplementedError()


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
