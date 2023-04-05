# This package contains all rankers

from experimaestro import tqdm

from enum import Enum
from pathlib import Path
from typing import (
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Protocol,
    Tuple,
    Callable,
    TypeVar,
    Union,
    final,
    TYPE_CHECKING,
)
import torch
import torch.nn as nn

import numpy as np
from experimaestro import Param, Config, Meta, DataPath
from datamaestro_text.data.ir import (
    AdhocDocuments,
    AdhocDocumentStore,
)
from xpmir.letor import Device, Random
from xpmir.letor.batchers import Batcher
from xpmir.letor.context import TrainerContext
from xpmir.letor.optim import Module
from xpmir.letor.records import (
    Document,
    BaseRecords,
    PairwiseRecord,
    PairwiseRecords,
    ProductRecords,
    Query,
)
from xpmir.utils.utils import EasyLogger, easylog

if TYPE_CHECKING:
    from xpmir.evaluation import RetrieverFactory

logger = easylog()


class ScoredDocument:
    """A data structure which contains the id, the content(optional) and a
    calculated score for a document"""

    def __init__(self, docid: Optional[str], score: float, content: str = None):
        self.docid = docid
        self.score = score
        self.content = content

    def __repr__(self):
        return f"document({self.docid}, {self.score}, {self.content})"

    def __lt__(self, other):
        return self.score < other.score


class ScorerOutputType(Enum):
    REAL = 0
    """An unbounded scalar value"""

    LOG_PROBABILITY = 1
    """A log probability, bounded by 0"""

    PROBABILITY = 2
    """A probability, in ]0,1["""


class LearnableModel(Config):
    """All learnable models"""

    pass


class Scorer(Config, EasyLogger):
    """Query-document scorer

    A model able to give a score to a list of documents given a query
    """

    outputType: ScorerOutputType = ScorerOutputType.REAL
    """Determines the type of output scalar (log probability, probability, logit) """

    def initialize(self, random: Optional[np.random.RandomState]):
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


def documents_retriever(
    documents: AdhocDocuments,
    *,
    retrievers: "RetrieverFactory" = None,
    scorer: Scorer = None,
    **kwargs,
):
    """Helper function"""
    assert retrievers is not None, "The retrievers have not been given"
    assert scorer is not None, "The scorer has not been given"
    return scorer.getRetriever(retrievers(documents), **kwargs)


class RandomScorer(Scorer):
    """A random scorer"""

    random: Param[Random]
    """The random number generator"""

    def rsv(
        self, query: str, documents: Iterable[ScoredDocument], keepcontent=False
    ) -> List[ScoredDocument]:
        scoredDocuments = []
        random = self.random.state
        for doc in documents:
            scoredDocuments.append(ScoredDocument(doc.docid, random.random()))
        return scoredDocuments


class AbstractLearnableScorer(Scorer, Module):
    """Base class for all learnable scorer"""

    checkpoint: DataPath[Optional[Path]]
    """A checkpoint path from which the model should be loaded (or None otherwise)"""

    __call__ = nn.Module.__call__
    to = nn.Module.to

    def __init__(self):
        nn.Module.__init__(self)
        super().__init__()
        self._initialized = False

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

        if self.checkpoint is None:
            # Sets the current random seed
            if random is not None:
                seed = random.randint((2**32) - 1)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            self._initialize(random)
        else:
            logger.info("Loading model from path %s", self.checkpoint)
            path = Path(self.checkpoint)
            self._initialize(None)
            self.load_state_dict(torch.load(path / "model.pth"))

        self._initialized = True

        return self


class LearnableScorer(AbstractLearnableScorer):
    """Learnable scorer

    A scorer with parameters that can be learnt"""

    def __call__(self, inputs: "BaseRecords", info: Optional[TrainerContext]):
        """Computes the score of all (query, document) pairs

        Different subclasses can process the input more or
        less efficiently based on the `BaseRecords` instance (pointwise,
        pairwise, or structured)
        """
        raise NotImplementedError(f"forward in {self.__class__}")

    def rsv(
        self,
        query: str,
        documents: Union[List[ScoredDocument], ScoredDocument, str, List[str]],
        content=False,
    ) -> List[ScoredDocument]:
        if isinstance(documents, str):
            documents = [ScoredDocument(None, None, documents)]
        elif isinstance(documents[0], str):
            documents = [ScoredDocument(None, None, text) for text in documents]

        # Prepare the inputs and call the model
        inputs = ProductRecords()
        for doc in documents:
            assert doc.content is not None

        inputs.addQueries(Query(None, query))
        inputs.addDocuments(*[Document(d.docid, d.content, d.score) for d in documents])

        with torch.no_grad():
            scores = self(inputs, None).cpu().numpy()

        # Returns the scored documents
        scoredDocuments = []
        for i in range(len(documents)):
            scoredDocuments.append(
                ScoredDocument(
                    documents[i].docid,
                    float(scores[i]),
                    documents[i].content if content else None,
                )
            )

        return scoredDocuments


class DuoLearnableScorer(AbstractLearnableScorer):
    """Base class for models that can score a triplet (query, document 1, document 2)"""

    def forward(self, inputs: "PairwiseRecords", info: Optional[TrainerContext]):
        """Returns scores for pairs of documents (given a query)"""
        raise NotImplementedError(f"abstract __call__ in {self.__class__}")


class Retriever(Config):
    """A retriever is a model to return top-scored documents given a query"""

    store: Param[Optional[AdhocDocumentStore]] = None
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

    def retrieve(self, query: str, content=False) -> List[ScoredDocument]:
        """Retrieves documents, returning a list sorted by decreasing score

        if `content` is true, includes the document full text
        """
        raise NotImplementedError()

    def _store(self) -> Optional[AdhocDocumentStore]:
        """Returns the associated document store (if any) that can be
        used to get the full text of the documents"""

    def get_store(self) -> Optional[AdhocDocumentStore]:
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
        content: bool,
    ):
        scoredDocuments.extend(self.scorer.rsv(query, batch, content))

    def retrieve(self, query: str, content=False):
        # Calls the retriever
        scoredDocuments = self.retriever.retrieve(query, content=True)

        # Scorer in evaluation mode
        self.scorer.eval()

        _scoredDocuments = []
        scoredDocuments = self._batcher.process(
            scoredDocuments, self._retrieve, query, _scoredDocuments, content
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
        scoredDocuments_previous = self.retriever.retrieve(query, content=True)

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
                    scoredDocuments_previous[i].docid, float(_scores_per_document[i])
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
        qry = Query(None, query)
        inputs = PairwiseRecords()
        for doc1, doc2 in documents:
            doc1 = Document(doc1.docid, doc1.content, doc1.score)
            doc2 = Document(doc2.docid, doc2.content, doc2.score)
            inputs.add(PairwiseRecord(qry, doc1, doc2))

        with torch.no_grad():
            scores = self.scorer(inputs, None).cpu().float()  # shape (batchsizes)
            return scores.tolist()


ARGS = TypeVar("ARGS")
KWARGS = TypeVar("KWARGS")
T = TypeVar("T")


class DocumentsFunction(Protocol, Generic[KWARGS, ARGS, T]):
    def __call__(self, documents: AdhocDocuments, *args: ARGS, **kwargs: KWARGS) -> T:
        ...


def document_cache(fn: DocumentsFunction[KWARGS, ARGS, T]):
    """Cache"""
    retrievers = {}

    def _fn(*args: ARGS, **kwargs: KWARGS):
        def cached(documents: AdhocDocuments) -> T:
            dataset_id = documents.__identifier__().all

            if dataset_id not in retrievers:
                retrievers[dataset_id] = fn(documents, *args, **kwargs)

            return retrievers[dataset_id]

        return cached

    return _fn


class ParametricRetrieverFactory(Protocol, Generic[KWARGS]):
    def __call__(**kwargs: KWARGS) -> Retriever:
        ...


class CollectionBasedRetrievers(Generic[KWARGS]):
    """Handles various retrievers depending on the collection"""

    def __init__(
        self,
        generator: Callable[
            [AdhocDocuments], Union[ParametricRetrieverFactory, Retriever]
        ],
    ):
        """Given a document collection, returns a factory"""
        self.generator = generator

    def factory(self, **kwargs) -> "RetrieverFactory":
        """Returns a retriever factory

        Caches the retrievers based on the document collection"""
        retrievers = {}

        def _factory(documents: AdhocDocuments):
            dataset_id = documents.__identifier__().all

            if dataset_id not in retrievers:
                retrievers[dataset_id] = self.generator(documents)

            if callable(retrievers[dataset_id]):
                return retrievers[dataset_id](**kwargs)

            assert not kwargs, f"{kwargs}"
            return retrievers[dataset_id]

        return _factory


def collection_based_retrievers(
    generator: Callable[[AdhocDocuments], Union[ParametricRetrieverFactory, Retriever]]
) -> CollectionBasedRetrievers:
    """Decorator for collection-dependent retriever factories

    Example of use

    .. highlight:: python
    .. code-block:: python

        @collection_based_retrievers
        def retrievers(documents: AdhocDocuments):
            index = IndexCollection(documents=documents).submit(launcher=launcher_index)
            return lambda *, k: RetrieverHydrator(store=documents,
                retriever=AnseriniRetriever(index=index, k=k, model=basemodel)
            )

        # Then this can be used to get
        test_retrievers = retrievers.factory(k=100)

    :param generator: A function that, given a document collection, should return
        either a retriever or a callable that returns a retriever
    :return: a collection based retriever
    """
    return CollectionBasedRetrievers(generator)


class RetrieverHydrator(Retriever):
    """Hydrate retrieved results with document text"""

    retriever: Param[Retriever]
    """The retriever to hydrate"""

    store: Param[AdhocDocumentStore]
    """The store for document texts"""

    def initialize(self):
        return self.retriever.initialize()

    def retrieve(self, query: str, content=False) -> List[ScoredDocument]:
        results = self.retriever.retrieve(query, content)

        if content:
            for result in results:
                result.content = self.store.document_text(result.docid)

        return results
