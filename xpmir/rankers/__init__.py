# This package contains all rankers

from logging import Logger
from typing import Iterable, List, Tuple
from experimaestro import Param, Config
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


class Scorer(Config, EasyLogger):
    """A model able to give a score to a list of documents given a query"""

    def rsv(
        self, query: str, documents: Iterable[ScoredDocument], keepcontent=False
    ) -> List[ScoredDocument]:
        """Score all the documents (inference mode, no training)"""
        raise NotImplementedError()


class RandomScorer(Scorer):
    """A random scorer

    Attributes:

    random: The random state
    """

    random: Param[Random]

    def rsv(
        self, query: str, documents: Iterable[ScoredDocument], keepcontent=False
    ) -> List[ScoredDocument]:
        scoredDocuments = []
        random = self.random.state
        for doc in documents:
            scoredDocuments.append(ScoredDocument(doc.docid, random.random()))
        return scoredDocuments


class LearnableScorer(Scorer):
    """A scorer with parameters that can be learnt"""

    pass


class Retriever(Config):
    """A retriever is a model to return top-scored documents given a query

    Attributes:

    topk: Number of documents to retrieve (used after re-reranking by the second stage retriever)
    """

    topk: Param[int] = 1500

    def initialize(self):
        pass

    def retrieve(self, query: str) -> List[ScoredDocument]:
        """Retrieves a documents, returning a list sorted by decreasing score"""
        raise NotImplementedError()

    def getindex(self) -> Index:
        """Returns the associated index (if any)"""
        raise NotImplementedError()


class TwoStageRetriever(Retriever):
    """Use on retriever to select the top-K documents which are the re-ranked given a scorer

    Attributes:

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
            # Work per batch
            _scoredDocuments = []
            for i in range(0, len(scoredDocuments), self.batchsize):
                _scoredDocuments.extend(
                    self.scorer.rsv(query, scoredDocuments[i : (i + self.batchsize)])
                )
        else:
            # Score everything at once
            _scoredDocuments = self.scorer.rsv(query, scoredDocuments)

        _scoredDocuments.sort(reverse=True)
        return _scoredDocuments[: self.topk]
