from typing import Iterator, List, NamedTuple, Optional
from datamaestro_text.data.ir import Adhoc
from experimaestro import config, param
import numpy as np
from xpmir.rankers import Retriever


class SamplerRecord(NamedTuple):
    query: str
    docid: str
    score: float
    relevance: Optional[float]


class Records:
    """Records are the objects passed to the module forwards"""

    # The queries
    queries: List[str]

    docids: List[str]
    scores: List[float]
    relevances: List[float]

    # Full text documents
    documents: Optional[List[str]]

    # Tokenized
    queries_toks: any
    docs_toks: any

    # Lengths (in tokens)
    queries_len: any
    docs_len: any

    # IDs of tokenized version
    queries_tokids: any
    docs_tokids: any

    def __init__(self):
        self.queries = []
        self.docids = []
        self.scores = []
        self.relevances = []

    def add(self, record: SamplerRecord):
        self.queries.append(record.query)
        self.docids.append(record.docid)
        self.relevances.append(record.relevance)
        self.scores.append(record.score)


@config()
class Collection:
    """Access to a document collection"""

    pass


@config()
class Sampler:
    """"Abtract data sampler"""

    def initialize(self, random: np.random.RandomState):
        self.random = random

    def record_iter(self) -> Iterator[SamplerRecord]:
        """Returns an iterator over records (query, document, relevance)
        """
        raise NotImplementedError()


@param("dataset", type=Adhoc, help="The topics and qrels")
@param("retriever", type=Retriever, help="The retriever")
@config()
class ModelBasedSampler(Sampler):
    def initialize(self, random):
        super().initialize(random)

        self.retriever.initialize()

        # Read the assessments
        assessments = {}
        for qrels in self.dataset.assessments.iter():
            doc2rel = {}
            assessments[qrels.qid] = doc2rel
            for docid, rel in qrels.assessments:
                doc2rel[docid] = rel

        # Read the topics
        self.records = []
        for query in self.dataset.topics.iter():
            qassessments = assessments.get(query.qid, None) or {}
            for sd in self.retriever.retrieve(query.title):
                rel = qassessments.get(sd.docid, 0)
                self.records.append(SamplerRecord(query.title, sd.docid, sd.score, rel))

    def record_iter(self) -> Iterator[SamplerRecord]:
        # FIXME: should not shuffle in place
        self.random.shuffle(self.records)
        return self.records
