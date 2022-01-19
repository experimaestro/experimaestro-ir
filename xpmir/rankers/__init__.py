# This package contains all rankers

from experimaestro import tqdm
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, final
import torch
import numpy as np
from experimaestro import Param, Config, Option, documentation, Meta
from datamaestro_text.data.ir import AdhocDocument, AdhocIndex as Index, AdhocDocuments
from xpmir.letor import Device, DeviceInformation, Random
from xpmir.letor.batchers import Batcher
from xpmir.letor.context import TrainerContext
from xpmir.letor.records import Document, BaseRecords, ProductRecords, Query
from xpmir.utils import EasyLogger, easylog

logger = easylog()


class ScoredDocument:
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


class Scorer(Config, EasyLogger):
    """Query-document scorer

    A model able to give a score to a list of documents given a query"""

    outputType: ScorerOutputType = ScorerOutputType.REAL

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


class LearnableScorer(Scorer):
    """Learnable scorer

    A scorer with parameters that can be learnt"""

    checkpoint: Meta[Optional[Path]]
    """A checkpoint path from which the model should be loaded (or None otherwise)"""

    def __init__(self):
        super().__init__()
        self._initialized = False

    def _initialize(self, random):
        raise NotImplementedError(f"_initialize in {self.__class__}")

    def train(self, mode=True):
        """Put the model in training mode"""
        raise NotImplementedError("train() in {self.__class__}")

    def eval(self):
        """Put the model in training mode"""
        self.train(False)

    def to(self, device):
        pass

    @final
    def initialize(self, random: Optional[np.random.RandomState]):
        """Initialize a learnable scorer

        Initialization can either be determined by a checkpoint (if set) or
        otherwise (random or pre-trained checkpoint depending on the models)
        """
        if self._initialized:
            return

        if self.checkpoint is None:
            # Sets the current random seed
            if random is not None:
                seed = random.randint((2 ** 32) - 1)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            self._initialize(random)
        else:
            logger.info("Loading model from path %s", self.checkpoint)
            path = Path(self.checkpoint)
            self._initialize(None)
            self.load_state_dict(torch.load(path / "model.pth"))

        self._initialized = True

    def __call__(self, inputs: "BaseRecords", info: Optional[TrainerContext]):
        """Computes the score of all (query, document) pairs

        Different subclasses can process the input more or
        less efficiently based on the `BaseRecords` instance (pointwise,
        pairwise, or structured)
        """
        raise NotImplementedError(f"forward in {self.__class__}")

    def rsv(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
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
            scoredDocuments.append(ScoredDocument(documents[i].docid, float(scores[i])))

        return scoredDocuments


class Retriever(Config):
    """A retriever is a model to return top-scored documents given a query"""

    def initialize(self):
        pass

    def collection(self):
        """Returns the document collection object"""
        raise NotImplementedError()

    def retrieve_all(self, queries: Dict[str, str]) -> Dict[str, List[ScoredDocument]]:
        """Retrieves for a set of documents

        By default, iterate using `self.retrieve`, but this leaves some room open
        for optimization"""
        results = {}
        for key, text in tqdm(list(queries.items())):
            results[key] = self.retrieve(text)
        return results

    def retrieve(self, query: str, content=False) -> List[ScoredDocument]:
        """Retrieves a documents, returning a list sorted by decreasing score

        if `content` is true, includes the document full text
        """
        raise NotImplementedError()

    def getindex(self) -> Index:
        """Returns the associated index (if any)"""
        raise NotImplementedError()

    @documentation
    def getReranker(
        self, scorer: Scorer, batch_size: int, batcher: Batcher = Batcher(), device=None
    ):
        """Returns a two stage re-ranker from this retriever and a scorer

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
    batcher: Meta[Batcher] = Batcher()
    device: Meta[Optional[Device]] = None

    def initialize(self):
        self.retriever.initialize()
        self._batcher = self.batcher.initialize(self.batchsize)
        self.scorer.initialize(None)

        # Compute with the scorer
        if self.device is not None:
            self.scorer.to(self.device.value)

    def _retrieve(
        self,
        batch: List[ScoredDocument],
        query: str,
        scoredDocuments: List[ScoredDocument],
    ):
        scoredDocuments.extend(self.scorer.rsv(query, batch))

    def retrieve(self, query: str):
        # Calls the retriever
        scoredDocuments = self.retriever.retrieve(query, content=True)

        # Scorer in evaluation mode
        self.scorer.eval()

        _scoredDocuments = []
        scoredDocuments = self._batcher.process(
            scoredDocuments, self._retrieve, query, _scoredDocuments
        )

        _scoredDocuments.sort(reverse=True)
        return _scoredDocuments
