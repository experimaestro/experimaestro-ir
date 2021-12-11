from typing import List, Optional
import itertools
import torch
from experimaestro import Config, Param
from xpmir.neural import DualRepresentationTorchScorer
from xpmir.vocab.encoders import TextEncoder
from xpmir.letor.traininfo import ScalarMetric, TrainingInformation


class DualVectorRegularizer(Config):
    def __call__(self, info: TrainingInformation, queries, documents):
        raise NotImplementedError(f"__call__ in {self.__class__}")


class DualVectorScorer(DualRepresentationTorchScorer):
    """A scorer based on dual vectorial representations"""

    pass


class Dense(DualVectorScorer):
    encoder: Param[TextEncoder]
    query_encoder: Param[Optional[TextEncoder]]
    regularizer: Param[Optional[DualVectorRegularizer]]

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def initialize(self, random):
        super().initialize(random)
        self.encoder.initialize()
        if self.query_encoder:
            self.query_encoder.initialize()

    def score_pairs(self, queries, documents, info):
        scores = (queries @ documents.transpose(1, 2)).squeeze()
        if info is not None and self.regularizer is not None:
            self.regularizer(info, queries, documents)
        return scores

    def as_document_encoder(self):
        return DenseDocumentEncoder(scorer=self)

    def as_query_encoder(self):
        return DenseQueryEncoder(scorer=self)


class DenseBaseEncoder(TextEncoder):
    scorer: Param[Dense]


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
        if self.scorer.query_encoder is None:
            return self.scorer.encoder.dimension
        return self.scorer.query_encoder.dimension

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a matrix encoding the provided texts"""
        return self.scorer.encode_queries(texts)


class CosineDense(Dense):
    """Siamese model (cosine)

    Attributes:
        encoder: Document (and query) encoder
        query_encoder: Query encoder; if null, uses the document encoder
    """

    def encode_queries(self, texts):
        queries = (self.query_encoder or self.encoder)(texts)
        return queries / queries.norm(dim=1, keepdim=True)

    def encode_documents(self, texts):
        documents = self.encoder(texts)
        return documents / documents.norm(dim=1, keepdim=True)


class DotDense(Dense):
    """Dual model based on inner product

    Attributes:
        encoder: Document (and query) encoder
        query_encoder: Query encoder; if null, uses the document encoder
    """

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def parameters(self):
        if self.query_encoder:
            return itertools.chain(
                self.query_encoder.parameters(), self.encoder.parameters()
            )
        return self.encoder.parameters()

    def encode_queries(self, texts):
        return (self.query_encoder or self.encoder)(texts)

    def encode_documents(self, texts):
        return self.encoder(texts)


class FlopsRegularizer(DualVectorRegularizer):
    lambda_q: Param[float]
    lambda_d: Param[float]

    def __call__(self, info: TrainingInformation, queries, documents):
        q = queries.abs().mean(0)
        flops_q = (q.unsqueeze(1) @ q.unsqueeze(2)).sum() / len(q)

        d = queries.abs().mean(0)
        flops_d = (d.unsqueeze(1) @ d.unsqueeze(2)).sum() / len(d)

        flops = self.lambda_d * flops_d + self.lambda_q * flops_q
        info.addRegularizer("flops", flops)

        info.metrics.add(ScalarMetric("flops", flops.item(), len(q)))
        info.metrics.add(ScalarMetric("flops_q", flops_q.item(), len(q)))
        info.metrics.add(ScalarMetric("flops_d", flops_d.item(), len(d)))
