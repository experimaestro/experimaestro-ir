# This package contains all rankers

from experimaestro import tqdm
from enum import Enum
from typing import (
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
    final,
    TYPE_CHECKING,
)
import torch
import torch.nn as nn
import attrs
import numpy as np
from experimaestro import Param, Config, Meta
from datamaestro_text.data.ir import (
    Document,
    Documents,
    DocumentStore,
)
from datamaestro_text.data.ir.base import TextDocument, TextTopic
from xpmir.utils.utils import Initializable
from xpmir.letor import Device, Random
from xpmir.learning.batchers import Batcher
from xpmir.learning.context import TrainerContext
from xpmir.learning.optim import Module
from xpmir.letor.records import (
    ScoredDocumentRecord,
    TopicRecord,
    BaseRecords,
    PairwiseRecord,
    PairwiseRecords,
    ProductRecords,
)
from xpmir.utils.utils import EasyLogger, easylog

if TYPE_CHECKING:
    from xpmir.evaluation import RetrieverFactory

logger = easylog()


@attrs.define()
class ScoredDocument:
    """A data structure that associated a score with a document"""

    document: Document
    """The document"""

    score: float
    """The associated score"""

    def __repr__(self):
        return f"document({self.document}, {self.score})"

    def __lt__(self, other):
        return self.score < other.score


class ScorerOutputType(Enum):
    REAL = 0
    """An unbounded scalar value"""

    LOG_PROBABILITY = 1
    """A log probability, bounded by 0"""

    PROBABILITY = 2
    """A probability, in ]0,1["""


class Scorer(Config, Initializable, EasyLogger):
    """Query-document scorer

    A model able to give a score to a list of documents given a query
    """

    outputType: ScorerOutputType = ScorerOutputType.REAL
    """Determines the type of output scalar (log probability, probability, logit) """

    def __initialize__(self, random: Optional[np.random.RandomState]):
        """Initialize the scorer

        Arguments:

            random:
                Random state for random number generation; when random is None,
                this means that the state will be loaded from
                disk after initializations
        """
        pass

    def rsv(
        self, query: str, documents: Iterable[ScoredDocument], keepcontent=False
    ) -> List[ScoredDocument]:
        """Score all the documents (inference mode, no training)"""
        raise NotImplementedError()

    def eval(self):
        """Put the model in inference/evaluation mode"""
        pass

    def to(self, device):
        """Move the scorer to another device"""
        pass

    def getRetriever(
        self,
        retriever: "Retriever",
        batch_size: int,
        batcher: Batcher = Batcher(),
        top_k=None,
        device=None,
    ):
        """Returns a two stage re-ranker from this retriever and a scorer

        :param device: Device for the ranker or None if no change should be made

        :param batch_size: The number of documents in each batch

        :param top_k: Number of documents to re-rank (or None for all)
        """
        return TwoStageRetriever(
            retriever=retriever,
            scorer=self,
            batchsize=batch_size,
            batcher=batcher,
            device=device,
            top_k=top_k if top_k else None,
        )


def scorer_retriever(
    documents: Documents,
    *,
    retrievers: "RetrieverFactory",
    scorer: Scorer,
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
    return scorer.getRetriever(retrievers(documents), **kwargs)


class RandomScorer(Scorer):
    """A random scorer"""

    random: Param[Random]
    """The random number generator"""

    def rsv(
        self, query: str, scored_documents: Iterable[ScoredDocument]
    ) -> List[ScoredDocument]:
        result = []
        random = self.random.state
        for scored_document in scored_documents:
            result.append(ScoredDocument(scored_document.document, random.random()))
        return result


class AbstractModuleScorer(Scorer, Module):
    """Base class for all learnable scorer"""

    # Ensures basic operations are redirected to torch.nn.Module methods
    __call__ = nn.Module.__call__
    to = nn.Module.to
    train = nn.Module.train

    def __init__(self):
        self.logger.info("Initializing %s", self)
        nn.Module.__init__(self)
        super().__init__()
        self._initialized = False

    def __str__(self):
        return f"scorer {self.__class__.__qualname__}"

    def _initialize(self, random):
        raise NotImplementedError(f"_initialize in {self.__class__}")

    def train(self, mode=True):
        """Put the model in training mode"""
        return nn.Module.train(self, mode)

    def eval(self):
        """Put the model in training mode"""
        self.train(False)

    @final
    def initialize(self, random: Optional[np.random.RandomState] = None):
        """Initialize a learnable scorer

        Initialization can either be determined by a checkpoint (if set) or
        otherwise (random or pre-trained checkpoint depending on the models)
        """
        if self._initialized:
            return

        # Sets the current random seed
        if random is not None:
            seed = random.randint((2**32) - 1)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self._initialize(random)
        else:
            self._initialize(None)

        self._initialized = True

        return self

    def rsv(
        self,
        query: str,
        scored_documents: Union[List[ScoredDocument], ScoredDocument, str, List[str]],
    ) -> List[ScoredDocument]:
        if isinstance(scored_documents, str):
            scored_documents = [ScoredDocument(TextDocument(scored_documents), None)]
        elif isinstance(scored_documents[0], str):
            scored_documents = [
                ScoredDocument(TextDocument(scored_document), None)
                for scored_document in scored_documents
            ]

        # Prepare the inputs and call the model
        inputs = ProductRecords()
        inputs.add_topics(TopicRecord(TextTopic(query)))

        inputs.add_documents(
            *[ScoredDocumentRecord(sd.document, sd.score) for sd in scored_documents]
        )

        with torch.no_grad():
            scores = self(inputs, None).cpu().numpy()

        # Returns the scored documents
        scoredDocuments = []
        for i in range(len(scored_documents)):
            scoredDocuments.append(
                ScoredDocument(
                    scored_documents[i].document,
                    float(scores[i]),
                )
            )

        return scoredDocuments


class LearnableScorer(AbstractModuleScorer):
    """Learnable scorer

    A scorer with parameters that can be learnt"""

    def forward(self, inputs: "BaseRecords", info: Optional[TrainerContext]):
        """Computes the score of all (query, document) pairs

        Different subclasses can process the input more or
        less efficiently based on the `BaseRecords` instance (pointwise,
        pairwise, or structured)
        """
        raise NotImplementedError(f"forward in {self.__class__}")


class DuoLearnableScorer(LearnableScorer):
    """Base class for models that can score a triplet (query, document 1, document 2)"""

    def forward(self, inputs: "PairwiseRecords", info: Optional[TrainerContext]):
        """Returns scores for pairs of documents (given a query)"""
        raise NotImplementedError(f"abstract __call__ in {self.__class__}")


class Retriever(Config):
    """A retriever is a model to return top-scored documents given a query"""

    store: Param[Optional[DocumentStore]] = None
    """Give the document store associated with this retriever"""

    def initialize(self):
        pass

    def collection(self):
        """Returns the document collection object"""
        raise NotImplementedError()

    def retrieve_all(self, queries: Dict[str, str]) -> Dict[str, List[ScoredDocument]]:
        """Retrieves for a set of documents

        By default, iterate using `self.retrieve`, but this leaves some room open
        for optimization

        Args:

            queries: A dictionary where the key is the ID of the query, and the value
                is the text
        """
        results = {}
        for key, text in tqdm(list(queries.items())):
            results[key] = self.retrieve(text)
        return results

    def retrieve(self, query: str) -> List[ScoredDocument]:
        """Retrieves documents, returning a list sorted by decreasing score

        if `content` is true, includes the document full text
        """
        raise NotImplementedError()

    def _store(self) -> Optional[DocumentStore]:
        """Returns the associated document store (if any) that can be
        used to get the full text of the documents"""

    def get_store(self) -> Optional[DocumentStore]:
        return self.store or self._store()


class AbstractTwoStageRetriever(Retriever):
    """Abstract class for all two stage retrievers (i.e. scorers and duo-scorers)"""

    retriever: Param[Retriever]
    """The base retriever"""

    scorer: Param[Scorer]
    """The scorer used to re-rank the documents"""

    top_k: Param[Optional[int]] = None
    """The number of returned documents (if None, returns all the documents)"""

    batchsize: Meta[int] = 0
    """The batch size for the re-ranker"""

    batcher: Meta[Batcher] = Batcher()
    """How to provide batches of documents"""

    device: Meta[Optional[Device]] = None
    """Device on which the model is run"""

    def initialize(self):
        self.retriever.initialize()
        self._batcher = self.batcher.initialize(self.batchsize)
        self.scorer.initialize(None)

        # Compute with the scorer
        if self.device is not None:
            self.scorer.to(self.device.value)


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

    def retrieve(self, query: str):
        # Calls the retriever
        scoredDocuments = self.retriever.retrieve(query)

        # Scorer in evaluation mode
        self.scorer.eval()

        _scoredDocuments = []
        scoredDocuments = self._batcher.process(
            scoredDocuments, self._retrieve, query, _scoredDocuments
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

    def retrieve(self, query: str):
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
        self, query: str, documents: List[Tuple[ScoredDocument, ScoredDocument]]
    ) -> List[float]:
        """Given the query and documents in tuple
        return the score for each triplets
        """
        qry = TopicRecord(TextTopic(query))
        inputs = PairwiseRecords()
        for doc1, doc2 in documents:
            inputs.add(PairwiseRecord(qry, doc1, doc2))

        with torch.no_grad():
            scores = self.scorer(inputs, None).cpu().float()  # shape (batchsizes)
            return scores.tolist()


ARGS = TypeVar("ARGS")
KWARGS = TypeVar("KWARGS")
T = TypeVar("T")


class DocumentsFunction(Protocol, Generic[KWARGS, ARGS, T]):
    def __call__(self, documents: Documents, *args: ARGS, **kwargs: KWARGS) -> T:
        ...


def document_cache(fn: DocumentsFunction[KWARGS, ARGS, T]):
    """Decorator

    Allows to cache the result of a function that should depend
    on the document dataset ID
    """
    retrievers = {}

    def _fn(*args: ARGS, **kwargs: KWARGS):
        def cached(documents: Documents) -> T:
            dataset_id = documents.__identifier__().all

            if dataset_id not in retrievers:
                retrievers[dataset_id] = fn(documents, *args, **kwargs)

            return retrievers[dataset_id]

        return cached

    return _fn


class RetrieverHydrator(Retriever):
    """Hydrate retrieved results with document text"""

    retriever: Param[Retriever]
    """The retriever to hydrate"""

    store: Param[DocumentStore]
    """The store for document texts"""

    def initialize(self):
        return self.retriever.initialize()

    def retrieve(self, query: str) -> List[ScoredDocument]:
        return [
            ScoredDocument(self.store.document_ext(sd.document.get_id()), sd.score)
            for sd in self.retriever.retrieve(query)
        ]
