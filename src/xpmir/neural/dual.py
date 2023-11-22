from typing import List, Optional
import torch
from experimaestro import Param
from xpmir.distributed import DistributableModel
from xpmir.learning.batchers import Batcher
from xpmir.neural import DualRepresentationScorer
from xpmir.rankers import Retriever
from xpmir.utils.utils import easylog, foreach
from xpmir.text.encoders import TextEncoder
from xpmir.learning.context import Loss, TrainerContext, TrainingHook
from xpmir.learning.metrics import ScalarMetric

logger = easylog()


class DualVectorListener(TrainingHook):
    """Listener called with the (vectorial) representation of queries and
    documents

    The hook is called just after the computation of documents and queries
    representations.

    This can be used for logging purposes, but more importantly, to add
    regularization losses such as the :class:`FlopsRegularizer` regularizer.
    """

    def __call__(
        self, context: TrainerContext, queries: torch.Tensor, documents: torch.Tensor
    ):
        """Hook handler

        Args:
            context (TrainerContext): The training context
            queries (torch.Tensor): The query vectors
            documents (torch.Tensor): The document vectors

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError(f"__call__ in {self.__class__}")


class DualVectorScorer(DualRepresentationScorer):
    """A scorer based on dual vectorial representations"""

    pass


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
        return scores

    @property
    def _query_encoder(self):
        return self.query_encoder or self.encoder

    @classmethod
    def from_sentence_transformers(cls, hf_id: str, **kwargs):
        """Creates a dense model from a Sentence transformer

        The list can be found on HuggingFace
        https://huggingface.co/models?library=sentence-transformers

        :param hf_id: The HuggingFace ID
        """
        from xpmir.text.huggingface import SentenceTransformerTextEncoder

        encoder = SentenceTransformerTextEncoder(hf_id)
        return cls(encoder, **kwargs)


class CosineDense(Dense):
    """Dual model based on cosine similarity."""

    def encode_queries(self, texts):
        queries = (self.query_encoder or self.encoder)(texts)
        return queries / queries.norm(dim=1, keepdim=True)

    def encode_documents(self, texts):
        documents = self.encoder(texts)
        return documents / documents.norm(dim=1, keepdim=True)


class DotDense(Dense, DistributableModel):
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

    def getRetriever(
        self, retriever: "Retriever", batch_size: int, batcher: Batcher, device=None
    ):
        from xpmir.rankers.full import FullRetrieverRescorer

        return FullRetrieverRescorer(
            documents=retriever.documents,
            scorer=self,
            batchsize=batch_size,
            batcher=batcher,
            device=device,
        )


class FlopsRegularizer(DualVectorListener):
    r"""The FLOPS regularizer computes

    .. math::

        FLOPS(q,d) = \lambda_q FLOPS(q) + \lambda_d FLOPS(d)

    where

    .. math::
        FLOPS(x) = \left( \frac{1}{d} \sum_{i=1}^d |x_i| \right)^2
    """
    lambda_q: Param[float]
    """Lambda for queries"""

    lambda_d: Param[float]
    """Lambda for documents"""

    def compute(x: torch.Tensor):
        """
        :param x: term vectors (batch size x vocabulary dimension)
        :returns: A couple (vocabulary dimension / FLOPS regularizations)
        """
        # Computes the mean for each term (weights are positive)
        y = x.mean(0)

        # Returns the sum of squared means
        return y, (y * y).sum()

    def __call__(self, info: TrainerContext, queries, documents):
        # queries and documents are length x dimension
        # Assumes that all weights are positive
        assert info.metrics is not None

        # q of shape (dimension), flops_q of shape (1)
        q, flops_q = FlopsRegularizer.compute(queries)
        d, flops_d = FlopsRegularizer.compute(documents)

        flops = self.lambda_d * flops_d + self.lambda_q * flops_q
        info.add_loss(Loss("flops", flops, 1.0))

        info.metrics.add(ScalarMetric("flops", flops.item(), 1))
        info.metrics.add(ScalarMetric("flops_q", flops_q.item(), 1))
        info.metrics.add(ScalarMetric("flops_d", flops_d.item(), 1))

        with torch.no_grad():
            info.metrics.add(
                ScalarMetric(
                    "sparsity_q",
                    torch.count_nonzero(queries).item()
                    / (queries.shape[0] * queries.shape[1]),
                    len(q),
                )
            )
            info.metrics.add(
                ScalarMetric(
                    "sparsity_d",
                    torch.count_nonzero(documents).item()
                    / (documents.shape[0] * documents.shape[1]),
                    len(d),
                )
            )


class ScheduledFlopsRegularizer(FlopsRegularizer):
    """
    The FLOPS regularizer where the lamdba_q and lambda_d varie according to the
    steps. The lambda values goes quadratic before the
    ```lamdba_warmup_steps```, and then remains constant
    """

    min_lambda_q: Param[float] = 0
    """Min value for the lambda_q before it increase"""

    min_lambda_d: Param[float] = 0
    """Min value for the lambda_d before it increase"""

    lamdba_warmup_steps: Param[int] = 0
    """The warmup steps for the lambda"""

    def quadratic_ratio(self, step):
        if step > self.lamdba_warmup_steps:
            return 1
        else:
            return (step / self.lamdba_warmup_steps) ** 2

    def __post_init__(self):
        self.initial_lambda_q = self.lambda_q
        self.initial_lambda_d = self.lambda_d

    def __call__(self, info: TrainerContext, queries, documents):
        current_step = info.steps
        self.lambda_q = (
            self.initial_lambda_q - self.min_lambda_q
        ) * self.quadratic_ratio(current_step) + self.min_lambda_q
        self.lambda_d = (
            self.initial_lambda_d - self.min_lambda_d
        ) * self.quadratic_ratio(current_step) + self.min_lambda_d
        FlopsRegularizer.__call__(self, info, queries, documents)
