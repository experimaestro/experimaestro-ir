from typing import Callable, List, Optional, Tuple
import itertools
import torch
from experimaestro import Config, Param, Meta
from xpmir.letor import DEFAULT_DEVICE, Device
from xpmir.neural import DualRepresentationScorer
from xpmir.utils import easylog, foreach
from xpmir.text.encoders import TextEncoder
from xpmir.letor.context import Loss, TrainerContext, TrainingHook
from xpmir.letor.metrics import ScalarMetric

logger = easylog()


class DualVectorListener(TrainingHook):
    """Regularizer called with the (vectorial) representation of queries and documents"""

    def __call__(
        self, info: TrainerContext, queries: torch.Tensor, documents: torch.Tensor
    ):
        raise NotImplementedError(f"__call__ in {self.__class__}")


class DualVectorScorer(DualRepresentationScorer):
    """A scorer based on dual vectorial representations"""


class Dense(DualVectorScorer):
    """A scorer based on a pair of (query, document) dense vectors"""

    encoder: Param[TextEncoder]
    """The document (and potentially query) encoder"""

    query_encoder: Param[Optional[TextEncoder]]
    """The query encoder (optional, if not defined uses the query_encoder)"""

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def _initialize(self, random):
        self.encoder.initialize()
        if self.query_encoder:
            self.query_encoder.initialize()

    def score_product(self, queries, documents, info: Optional[TrainerContext]):
        return queries @ documents.T

    def score_pairs(self, queries, documents, info: TrainerContext):
        scores = (queries.unsqueeze(1) @ documents.unsqueeze(2)).squeeze(-1).squeeze(-1)

        # Apply the dual vector hook
        if info is not None:
            foreach(
                info.hooks(DualVectorListener),
                lambda hook: hook(info, queries, documents),
            )
        (queries, documents)
        return scores

    @property
    def _query_encoder(self):
        return self.query_encoder or self.encoder


class DenseBaseEncoder(TextEncoder):
    """A text encoder adapter for dense scorers (either query or document encoder)"""

    scorer: Param[Dense]

    def initialize(self):
        self.scorer.initialize(None)


class DenseDocumentEncoder(DenseBaseEncoder):
    @property
    def dimension(self):
        """Returns the dimension of the representation"""
        return self.scorer.encoder.dimension

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a matrix encoding the provided texts"""
        return self.scorer.encode_documents(texts)


class DenseQueryEncoder(DenseBaseEncoder):
    @property
    def dimension(self):
        """Returns the dimension of the representation"""
        return self.scorer._query_encoder.dimension

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a matrix encoding the provided texts"""
        return self.scorer.encode_queries(texts)


class CosineDense(Dense):
    """Dual model based on cosine similarity."""

    def encode_queries(self, texts):
        queries = (self.query_encoder or self.encoder)(texts)
        return queries / queries.norm(dim=1, keepdim=True)

    def encode_documents(self, texts):
        documents = self.encoder(texts)
        return documents / documents.norm(dim=1, keepdim=True)


class DotDense(Dense):
    """Dual model based on inner product."""

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def encode_queries(self, texts: List[str]):
        """Encode the different queries"""
        return self._query_encoder(texts)

    def encode_documents(self, texts: List[str]):
        """Encode the different documents"""
        return self.encoder(texts)


class FlopsRegularizer(DualVectorListener):
    lambda_q: Param[float]
    lambda_d: Param[float]
    weight: Param[float] = 1.0

    @staticmethod
    def compute(x: torch.Tensor):
        # Computes the mean for each term
        y = x.abs().mean(0)
        # Returns the sum of squared means
        return y, (y * y).sum()

    def __call__(self, info: TrainerContext, queries, documents):
        # queries and documents are length x dimension
        # Assumes that all weights are positive

        q, flops_q = FlopsRegularizer.compute(queries)
        d, flops_d = FlopsRegularizer.compute(documents)

        flops = self.lambda_d * flops_d + self.lambda_q * flops_q
        info.add_loss(Loss("flops", flops, self.weight))

        info.metrics.add(ScalarMetric("flops", flops.item(), len(q)))
        info.metrics.add(ScalarMetric("flops_q", flops_q.item(), len(q)))
        info.metrics.add(ScalarMetric("flops_d", flops_d.item(), len(d)))

        with torch.no_grad():
            info.metrics.add(
                ScalarMetric(
                    "sparsity_q",
                    (queries != 0).sum().item() / (queries.shape[0] * queries.shape[1]),
                    len(q),
                )
            )
            info.metrics.add(
                ScalarMetric(
                    "sparsity_d",
                    (documents != 0).sum().item()
                    / (documents.shape[0] * documents.shape[1]),
                    len(d),
                )
            )
