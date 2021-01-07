# This package contains all rankers

from logging import Logger
from typing import List, Tuple
from experimaestro import config, param
from xpmir.letor import Random
from xpmir.utils import EasyLogger


class ScoredDocument:
    def __init__(self, docid: str, score: float):
        self.docid = docid
        self.score = score

    def __lt__(self, other):
        return other.score < self.score


@config()
class Scorer(EasyLogger):
    """A model able to give a score to a list of documents given a query"""

    def rsv(
        self, query: str, docids: List[str], scores: List[float]
    ) -> List[ScoredDocument]:
        raise NotImplementedError()


@param("random", type=Random, help="Random state")
@config()
class RandomScorer(Scorer):
    """A random scorer"""

    def rsv(
        self, query: str, docids: List[str], scores: List[float]
    ) -> List[ScoredDocument]:
        l = []
        random = self.random.state
        for docid in docids:
            l.append(ScoredDocument(docid, random.random()))
        return l


@config()
class LearnableScorer(Scorer):
    pass


@param("topk", type=int, default=1500, help="Number of documents to retrieve")
@config()
class Retriever:
    """A retriever is a model able to retrieve"""

    def initialize(self):
        pass

    def retrieve(query: str) -> List[ScoredDocument]:
        raise NotImplementedError()


@param("retriever", type=Retriever, help="The base retriever")
@param("scorer", type=Scorer, help="The scorer used to re-rank the documents")
@config()
class TwoStageRetriever(Retriever):
    def initialize(self):
        self.retriever.initialize()

    def retrieve(self, query: str):
        scoredDocuments = self.retriever.retrieve(query)

        scoredDocuments = self.scorer.rsv(
            query,
            [sd.docid for sd in scoredDocuments],
            [sd.score for sd in scoredDocuments],
        )
        scoredDocuments.sort(reverse=True)
        return scoredDocuments[: self.topk]
