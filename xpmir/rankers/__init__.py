# This package contains all rankers

from typing import List, Tuple
from experimaestro import config, param


@config()
class Scorer:
    """A model able to give a score to a document given a query"""

    def rsv(self, query, documents) -> Tuple[List[str], List[float]]:
        raise NotImplementedError()


@config()
class LearnableScorer(Scorer):
    pass


class ScoredDocument:
    def __init__(self, docid: str, score: float):
        self.docid = docid
        self.score = score

    def __lt__(self, other):
        return other.score < self.score


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
        scoredDocuments = self.scorer.rsv(query, [sd.docid for sd in scoredDocuments])
        scoredDocuments.sort(reverse=True)
        return scoredDocuments[: self.topk]
