# This package contains all rankers
from abc import ABC, abstractmethod
from typing import (
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    TYPE_CHECKING,
)
import torch
import torch.nn as nn
from experimaestro import Param, Config, Meta, field
from datamaestro_ir.data import (
    Documents,
    IDTextRecord,
    SimpleTextItem,
)
from xpm_torch import Module, Random
from xpm_torch.utils.utils import Initializable
from xpm_torch.utils.logging import EasyLogger
from xpm_torch.batchers import Batcher
from xpm_torch.learner import TrainerContext
from xpm_torch.losses import ModuleOutputType
from xpmir.letor.records import (
    BaseItems,
    PairwiseItem,
    PairwiseItems,
    ProductItems,
)
from datamaestro_ir.data.base import ScoredDocument
from .retriever import Retriever

import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from xpmir.evaluation import RetrieverFactory


class Scorer(Config, Initializable, EasyLogger, ABC):
    """Query-document scorer

    A model able to give a score to a list of documents given a query
    """

    outputType: ModuleOutputType = ModuleOutputType.REAL
    """Determines the type of output scalar (log probability, probability, logit) """

    doc: Meta[str] = ""
    """Paper description or title (used in HF Hub README)"""

    bibtex: Meta[str] = ""
    """BibTeX citation (used in HF Hub README)"""

    def __initialize__(self):
        """Initialize the scorer"""
        pass

    def rsv(
        self,
        topic: Union[str, IDTextRecord],
        documents: Union[List[ScoredDocument], ScoredDocument, str, List[str]],
    ) -> List[ScoredDocument]:
        # Convert into document records
        if isinstance(documents, str):
            documents = [ScoredDocument({"text_item": SimpleTextItem(documents)}, None)]
        elif isinstance(documents[0], str):
            documents = [
                ScoredDocument({"text_item": SimpleTextItem(scored_document)}, None)
                for scored_document in documents
            ]

        # Convert into topic record
        if isinstance(topic, str):
            topic = {"text_item": SimpleTextItem(topic)}

        return self.compute(topic, documents)

    @abstractmethod
    def compute(
        self, topic: IDTextRecord, documents: Iterable[ScoredDocument]
    ) -> List[ScoredDocument]:
        """Score all documents with respect to the topic"""
        ...

    def getRetriever(
        self,
        retriever: "Retriever",
        batch_size: int,
        batcher: Batcher = Batcher.C(),
        top_k=None,
        device=None,
    ):
        """Returns a two stage re-ranker from this retriever and a scorer

        :param device: Device for the ranker or None if no change should be made

        :param batch_size: The number of documents in each batch

        :param top_k: Number of documents to re-rank (or None for all)
        """
        return TwoStageRetriever.C(
            retriever=retriever,
            scorer=self,
            batchsize=batch_size,
            batcher=batcher,
            top_k=top_k if top_k else None,
        )


def scorer_retriever(
    documents: Documents,
    *,
    retrievers: "RetrieverFactory",
    scorer: Scorer,
    key: str = None,
    **kwargs,
):
    """Helper function that returns a two stage retriever. This is useful
    when used with partial (when the scorer is not known).

    :param documents: The document collection
    :param retrievers: A retriever factory
    :param scorer: The scorer
    :return: A retriever, calling the :meth:scorer.getRetriever
    """
    assert retrievers is not None, "The retrievers have not been given"
    assert scorer is not None, "The scorer has not been given"
    return scorer.getRetriever(retrievers(documents, key=key), **kwargs)


class RandomScorer(Scorer):
    """A random scorer"""

    random: Param[Random]
    """The random number generator"""

    def compute(
        self, record: IDTextRecord, scored_documents: Iterable[ScoredDocument]
    ) -> List[ScoredDocument]:
        result = []
        random = self.random.state
        for scored_document in scored_documents:
            result.append(ScoredDocument(scored_document.document, random.random()))
        return result


class AbstractModuleScorerCall(Protocol):
    def __call__(self, inputs: "BaseItems", info: Optional[TrainerContext]): ...


class AbstractModuleScorer(Scorer, Module):
    """Base class for all torch-based Modules implementing the `xpmir.rankers.Scorer`

    This class provides a `compute` method that calls the forward method,

    """

    # Ensures basic operations are redirected to torch.nn.Module methods
    __call__: AbstractModuleScorerCall = nn.Module.__call__
    train = nn.Module.train

    def __init__(self):
        logger.info(f"Initializing {self.__class__.__name__}")
        nn.Module.__init__(self)
        super().__init__()
        self._initialized = False

    def __initialize__(self):
        """Initialize a learnable scorer (structure only)"""
        return self

    def compute(
        self, topic: IDTextRecord, scored_documents: Iterable[ScoredDocument]
    ) -> List[ScoredDocument]:
        # Prepare the inputs and call the model
        inputs = ProductItems()
        inputs.add_topics(topic)

        inputs.add_documents(*[sd.document for sd in scored_documents])

        with torch.no_grad():
            scores = self(inputs, None).cpu().float().numpy()

        # Returns the scored documents
        scoredDocuments = []
        for i in range(len(scored_documents)):
            scoredDocuments.append(
                ScoredDocument(
                    scored_documents[i].document,
                    float(scores[i].item()),
                )
            )

        return scoredDocuments


class DuoLearnableScorer(AbstractModuleScorer):
    """Base class for models that can score a triplet (query, document 1, document 2)"""

    def forward(self, inputs: "PairwiseItems", info: Optional[TrainerContext]):
        """Returns scores for pairs of documents (given a query)"""
        raise NotImplementedError(f"abstract __call__ in {self.__class__}")


class AbstractTwoStageRetriever(Retriever):
    """Abstract class for all two stage retrievers (i.e. scorers and duo-scorers)"""

    retriever: Param[Retriever]
    """The base retriever"""

    scorer: Param[Scorer]
    """The scorer used to re-rank the documents"""

    top_k: Param[Optional[int]]
    """The number of returned documents (if None, returns all the documents)"""

    batchsize: Meta[int] = field(default=0, ignore_default=True)
    """The batch size for the re-ranker"""

    batcher: Meta[Batcher] = field(default_factory=Batcher.C)
    """How to provide batches of documents"""

    def initialize(self):
        self.retriever.initialize()
        self._batcher = self.batcher.initialize(self.batchsize)
        self.scorer.initialize()


class TwoStageRetriever(AbstractTwoStageRetriever):
    """Use on retriever to select the top-K documents which are the re-ranked
    given a scorer"""

    def _retrieve(
        self,
        batch: List[ScoredDocument],
        query: str,
        scoredDocuments: List[ScoredDocument],
    ):
        scoredDocuments.extend(self.scorer.rsv(query, batch))

    def retrieve(self, record: IDTextRecord):
        # Calls the retriever
        scoredDocuments = self.retriever.retrieve(record)

        # Scorer in evaluation mode
        self.scorer.eval()

        _scoredDocuments = []
        scoredDocuments = self._batcher.process(
            scoredDocuments, self._retrieve, record, _scoredDocuments
        )

        _scoredDocuments.sort(reverse=True)
        return _scoredDocuments[: (self.top_k or len(_scoredDocuments))]


class DuoTwoStageRetriever(AbstractTwoStageRetriever):
    """The two stage retriever for pairwise scorers.

    For pairwise scorer, we need to aggregate the pairwise scores in some
    way.
    """

    def _retrieve(
        self,
        batch: List[Tuple[ScoredDocument, ScoredDocument]],
        query: str,
        scoredDocuments: List[float],
    ):
        """call the function rsv to get the information for each batch
        because of the batchsize is independent on k, we may seperate the
        triplets belongs to the same query into different batches.
        """
        scoredDocuments.extend(self.rsv(query, batch))

    def retrieve(self, query: IDTextRecord):
        """call the _retrieve function by using the batcher and do an
        aggregation of all the scores
        """
        # get the documents from the retriever
        scoredDocuments_previous = self.retriever.retrieve(query)

        # transform them into the pairs (i, j)
        # for i != j ranging from 1 to nb of documents
        pairs = []
        for i in range(len(scoredDocuments_previous)):
            for j in range(len(scoredDocuments_previous)):
                if i != j:
                    pairs.append(
                        (scoredDocuments_previous[i], scoredDocuments_previous[j])
                    )

        # Scorer in evaluation mode
        self.scorer.eval()

        _scores_pairs = []  # the scores for each pair of documents
        self._batcher.process(pairs, self._retrieve, query, _scores_pairs)

        # Use the sum aggregation strategy
        _scores_pairs = torch.Tensor(_scores_pairs).reshape(
            len(scoredDocuments_previous), -1
        )
        _scores_per_document = torch.sum(
            _scores_pairs, dim=1
        )  # scores for each document.

        # construct the ScoredDocument object from the score we just get.
        scoredDocuments = []
        for i in range(len(scoredDocuments_previous)):
            scoredDocuments.append(
                ScoredDocument(
                    scoredDocuments_previous[i], float(_scores_per_document[i])
                )
            )
        scoredDocuments.sort(reverse=True)
        return scoredDocuments[: (self.top_k or len(scoredDocuments))]

    def rsv(
        self,
        record: IDTextRecord,
        documents: List[Tuple[ScoredDocument, ScoredDocument]],
    ) -> List[float]:
        """Given the query and documents in tuple
        return the score for each triplets
        """
        inputs = PairwiseItems()
        for doc1, doc2 in documents:
            inputs.add(PairwiseItem(record, doc1, doc2))

        with torch.no_grad():
            scores = self.scorer(inputs, None).cpu().float()  # shape (batchsizes)
            return scores.tolist()
