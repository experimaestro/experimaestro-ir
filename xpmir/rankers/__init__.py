# This package contains all rankers

from logging import Logger
from typing import Iterator, List, Tuple
from experimaestro import config, param, Param
from xpmir.dm.data import Index
from xpmir.letor import Random
from xpmir.utils import EasyLogger


class ScoredDocument:
    def __init__(self, docid: str, score: float, content: str = None):
        self.docid = docid
        self.score = score
        self.content = content

    def __lt__(self, other):
        return self.score < other.score


@config()
class Scorer(EasyLogger):
    """A model able to give a score to a list of documents given a query"""

    def rsv(
        self, query: str, documents: Iterator[ScoredDocument], keepcontent=False
    ) -> List[ScoredDocument]:
        raise NotImplementedError()


@param("random", type=Random, help="Random state")
@config()
class RandomScorer(Scorer):
    """A random scorer"""

    def rsv(
        self, query: str, documents: Iterator[ScoredDocument], keepcontent=False
    ) -> List[ScoredDocument]:
        scoredDocuments = []
        random = self.random.state
        for doc in documents:
            scoredDocuments.append(ScoredDocument(doc.docid, random.random()))
        return scoredDocuments


@config()
class LearnableScorer(Scorer):
    pass


@config()
class Retriever:
    """A retriever is a model able to retrieve

    Attributes:
        topk: Number of documents to retrieve
    """

    topk: Param[int] = 1500

    def initialize(self):
        pass

    def retrieve(query: str) -> List[ScoredDocument]:
        """Retrieves a documents, returning a list sorted by decreasing score"""
        raise NotImplementedError()

    def index(self) -> Index:
        raise NotImplementedError()


@config()
class TwoStageRetriever(Retriever):
    """[summary]

    Args:
        retriever: The base retriever
        scorer: The scorer used to re-rank the documents
        batchsize: The batch size for the re-ranker
    """

    retriever: Param[Retriever]
    scorer: Param[Scorer]
    batchsize: Param[int] = 0

    def initialize(self):
        self.retriever.initialize()

    def retrieve(self, query: str):
        scoredDocuments = self.retriever.retrieve(query)

        if self.batchsize > 0:
            _scoredDocuments = []
            for i in range(0, len(scoredDocuments), self.batchsize):
                _scoredDocuments.extend(
                    self.scorer.rsv(query, scoredDocuments[i : (i + self.batchsize)])
                )
        else:
            _scoredDocuments = self.scorer.rsv(query, scoredDocuments)

        _scoredDocuments.sort(reverse=True)
        return _scoredDocuments[: self.topk]
