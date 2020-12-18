# This package contains all rankers

from typing import List, Tuple
from experimaestro import config, param


@config()
class Scorer:
    """A model able to give a score to a document given a query"""

    def rsv(query, documents) -> Tuple[List[str], List[float]]:
        pass


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

    def retrieve(query: str) -> List[ScoredDocument]:
        raise NotImplementedError()


@param("retriever", type=Retriever, help="The base retriever")
@param("scorer", type=Scorer, help="The scorer used to re-rank the documents")
@config()
class TwoStageRetriever(Retriever):
    def retrieve(self, query: str):
        scoredDocuments = self.retriever.retrieve(query)
        scoredDocuments = self.rsv(query, [sd.docid for sd in scoredDocuments])
        scoredDocuments.sort(reverse=True)
        return scoredDocuments[: self.topk]
