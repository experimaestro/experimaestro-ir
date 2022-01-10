from typing import Callable, List, Optional, Tuple
import itertools
import torch
from experimaestro import Config, Param, Meta
from xpmir.letor import DEFAULT_DEVICE, Device
from xpmir.neural import DualRepresentationScorer
from xpmir.utils import easylog, foreach
from xpmir.text.encoders import TextEncoder
from xpmir.letor.context import TrainerContext, TrainingHook
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

    pass


class Dense(DualVectorScorer):
    """A scorer based on a pair of (query, document) dense vectors"""

    encoder: Param[TextEncoder]
    query_encoder: Param[Optional[TextEncoder]]

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def _initialize(self, random):
        self.encoder.initialize()
        if self.query_encoder:
            self.query_encoder.initialize()

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
    """Dual model based on cosine similarity"""

    def encode_queries(self, texts):
        queries = (self.query_encoder or self.encoder)(texts)
        return queries / queries.norm(dim=1, keepdim=True)

    def encode_documents(self, texts):
        documents = self.encoder(texts)
        return documents / documents.norm(dim=1, keepdim=True)


class DotDense(Dense):
    """Dual model based on inner product"""

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def encode_queries(self, texts):
        return self._query_encoder(texts)

    def encode_documents(self, texts):
        return self.encoder(texts)


class FlopsRegularizer(DualVectorListener):
    lambda_q: Param[float]
    lambda_d: Param[float]

    def __call__(self, info: TrainerContext, queries, documents):
        q = queries.abs().mean(0)
        flops_q = (q.unsqueeze(1) @ q.unsqueeze(2)).sum()

        d = documents.abs().mean(0)
        flops_d = (d.unsqueeze(1) @ d.unsqueeze(2)).sum()

        flops = self.lambda_d * flops_d + self.lambda_q * flops_q
        info.add_loss(flops)

        info.metrics.add(ScalarMetric("flops", flops.item(), len(q)))
        info.metrics.add(ScalarMetric("flops_q", flops_q.item(), len(q)))
        info.metrics.add(ScalarMetric("flops_d", flops_d.item(), len(d)))

        with torch.no_grad():
            info.metrics.add(ScalarMetric("sparsity_q", (q != 0).mean().item(), len(q)))
            info.metrics.add(ScalarMetric("sparsity_d", (d != 0).mean().item(), len(d)))
