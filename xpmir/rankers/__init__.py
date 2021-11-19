# This package contains all rankers

from typing import Iterable, Iterator, List, Optional
import torch
from experimaestro import Param, Config, Option, documentation
from xpmir.index import Index
from xpmir.letor import DEFAULT_DEVICE, Device, Random
from xpmir.letor.batchers import Batcher
from xpmir.letor.records import Document, BaseRecords, Query
from xpmir.utils import EasyLogger, easylog

logger = easylog()


class ScoredDocument:
    def __init__(self, docid: str, score: float, content: str = None):
        self.docid = docid
        self.score = score
        self.content = content

    def __lt__(self, other):
        return self.score < other.score


class Scorer(Config, EasyLogger):
    """Query-document scorer

    A model able to give a score to a list of documents given a query"""

    def rsv(
        self, query: str, documents: Iterable[ScoredDocument], keepcontent=False
    ) -> List[ScoredDocument]:
        """Score all the documents (inference mode, no training)"""
        raise NotImplementedError()

    def eval(self):
        """Put the model in inference/evaluation mode"""
        pass

    def train(self):
        """Put the model in training mode"""
        pass


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
    """Learnable scorer

    A scorer with parameters that can be learnt"""

    def initialize(self, random_or_state):
        """Initialize a learnable scorer

        Args:
            random_or_state (np.random.Random): The random state or the current state
        """
        pass

    def __call__(self, inputs: "BaseRecords"):
        """Computes the score of all (query, document) pairs

        Different subclasses can process the input more or
        less efficiently based on the `BaseRecords` instance (pointwise,
        pairwise, or structured)
        """
        raise NotImplementedError(f"forward in {self.__class__}")

    def rsv(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
        # Prepare the inputs and call the model
        inputs = BaseRecords()
        for doc in documents:
            assert doc.content is not None

        inputs.queries = [Query(query) for _ in documents]
        inputs.documents = [Document(d.docid, d.content, d.score) for d in documents]

        with torch.no_grad():
            scores = self(inputs).cpu().numpy()

        # Returns the scored documents
        scoredDocuments = []
        for i in range(len(documents)):
            scoredDocuments.append(ScoredDocument(documents[i].docid, float(scores[i])))

        return scoredDocuments


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

    @documentation
    def getReranker(
        self, scorer: Scorer, batch_size: int, batcher: Batcher = Batcher(), device=None
    ):
        """Retrieves a two stage re-ranker

        Arguments:
            device: Device for the ranker or None if no change should be made
        """
        return TwoStageRetriever(
            retriever=self,
            scorer=scorer,
            batchsize=batch_size,
            batcher=batcher,
            device=device,
        )


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
    batcher: Param[Batcher] = Batcher()
    device: Option[Optional[Device]] = None

    def initialize(self):
        self.retriever.initialize()
        self._batcher = self.batcher.initialize(self.batchsize, self._retrieve)

        # Compute with the scorer
        if self.device is not None:
            self.scorer.to(self.device(logger))

    def _retrieve(self, batches: Iterator[List[ScoredDocument]], query: str):
        _scoredDocuments = []
        for batch in batches:
            _scoredDocuments.extend(self.scorer.rsv(query, batch))
        return _scoredDocuments

    def retrieve(self, query: str):
        # Calls the retriever
        scoredDocuments = self.retriever.retrieve(query)

        # Scorer in evaluation mode
        self.scorer.eval()

        scoredDocuments = self._batcher(scoredDocuments, query)

        scoredDocuments.sort(reverse=True)
        return scoredDocuments[: self.topk]
