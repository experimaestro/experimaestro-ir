import torch
import torch.nn as nn
from experimaestro import Param
from xpmir.letor.records import BaseRecords
from xpmir.neural import TorchLearnableScorer
from xpmir.vocab.encoders import DualTextEncoder


class JointClassifier(TorchLearnableScorer):
    """Query-Document Representation Classifier

    Based on a query-document representation representation (e.g. BERT [CLS] token),
    takes a decision

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

    def forward(self, inputs: BaseRecords):
        # Encode queries and documents
        pairs = self.encoder(
            [(q.text, d.text) for q, d in zip(inputs.queries, inputs.documents)]
        )
        return self.classifier(pairs).squeeze(1)
