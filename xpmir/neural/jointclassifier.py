from typing import Iterable, List, Optional
import itertools
import torch
import torch.nn as nn
from experimaestro import Config, Param
from xpmir.letor.records import PointwiseRecord, Records
from xpmir.rankers import LearnableScorer, ScoredDocument
from xpmir.vocab.encoders import DualTextEncoder


class JointClassifier(LearnableScorer, nn.Module):
    """Transformer-based classification based on the [CLS] token

    Attributes:
        encoder: Document (and query) encoder
        query_encoder: Query encoder; if null, uses the document encoder
    """

    encoder: Param[DualTextEncoder]

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

    def initialize(self, random):
        super().initialize(random)
        self.encoder.initialize()
        self.classifier = torch.nn.Linear(self.encoder.dimension, 1)

    def parameters(self):
        return self.encoder.parameters()

    def forward(self, inputs: Records):
        # Encode queries and documents
        pairs = self.encoder(
            [(q, d.text) for q, d in zip(inputs.queries, inputs.documents)]
        )
        return self.classifier(pairs).squeeze(1)
