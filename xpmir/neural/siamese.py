from typing import List
import itertools
from experimaestro import Config, Param
from xpmir.letor.samplers import Records
from xpmir.rankers import LearnableScorer


class TextEncoder(Config):
    @property
    def dimension(self):
        raise NotImplementedError()

    def __call__(self, texts: List[str]):
        raise NotImplementedError()


class CosineSiamese(LearnableScorer):
    """Siamese model (cosine)

    Attributes:

        compression_size: Projection layer for the last layer (or 0 if None)
    """

    query_encoder: Param[TextEncoder]
    document_encoder: Param[TextEncoder]

    def __validate__(self):
        super().__validate__()

    def initialize(self, random):
        super().initialize(random)

    def parameters(self):
        return itertools.chain(
            self.query_encoder.parameters(), self.document_encoder.parameters()
        )

    def forward(self, inputs: Records):
        # Encode queries and documents
        queries = self.query_encoder([d.text for d in inputs.documents])
        documents = self.document_encoder(inputs.queries)

        # Normalize each document and query
        queries = queries / queries.norm(dim=1, keepdim=True)
        documents = documents / documents.norm(dim=1, keepdim=True)

        # Compute batch dot product and return it
        return queries.unsqueeze(1) @ documents.unsqueeze(2)
