from typing import Optional
import itertools
import torch
from experimaestro import Param
from xpmir.letor.records import BaseRecords
from xpmir.neural import SeparateRepresentationTorchScorer
from xpmir.vocab.encoders import TextEncoder


class Dense(SeparateRepresentationTorchScorer):
    encoder: Param[TextEncoder]
    query_encoder: Param[Optional[TextEncoder]]

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def initialize(self, random):
        super().initialize(random)
        self.encoder.initialize()
        if self.query_encoder:
            self.query_encoder.initialize()


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
    """Siamese model (cosine)

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
