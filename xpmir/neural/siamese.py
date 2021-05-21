from typing import Iterable, List, Optional
import itertools
import torch
import torch.nn as nn
from experimaestro import Config, Param
from xpmir.letor.records import PointwiseRecord, Records
from xpmir.rankers import LearnableScorer, ScoredDocument


class TextEncoder(Config):
    @property
    def dimension(self):
        raise NotImplementedError()


class CosineSiamese(LearnableScorer, nn.Module):
    """Siamese model (cosine)

    Attributes:
        encoder: Document (and query) encoder
        query_encoder: Query encoder (or null)
    """

    encoder: Param[TextEncoder]
    query_encoder: Param[Optional[TextEncoder]]

    def __validate__(self):
        super().__validate__()

    def initialize(self, random):
        super().initialize(random)
        self.encoder.initialize()
        if self.query_encoder:
            self.query_encoder.initialize()

    def parameters(self):
        if self.query_encoder:
            return itertools.chain(
                self.query_encoder.parameters(), self.encoder.parameters()
            )
        return self.encoder.parameters()

    def forward(self, inputs: Records):
        # Encode queries and documents
        queries = (self.query_encoder or self.encoder)(
            [d.text for d in inputs.documents]
        )
        documents = self.encoder(inputs.queries)

        # Normalize each document and query
        queries = queries / queries.norm(dim=1, keepdim=True)
        documents = documents / documents.norm(dim=1, keepdim=True)

        # Compute batch dot product and return it
        scores = queries.unsqueeze(1) @ documents.unsqueeze(2)
        return scores.squeeze()
