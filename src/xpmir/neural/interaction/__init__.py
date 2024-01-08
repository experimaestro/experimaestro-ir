from typing import Optional
from experimaestro import Param
from xpmir.rankers import LearnableScorer
from xpmir.text import TokensEncoder
from xpmir.letor.records import BaseRecords
from xpmir.learning.context import TrainerContext


class InteractionScorer(LearnableScorer):
    """Interaction-based neural scorer

    This is the base class for all scorers that depend on a map
    of cosine/inner products between query and document token representations.
    """

    encoder: Param[TokensEncoder]
    """The embedding model -- the vocab also defines how to tokenize text"""

    query_encoder: Param[Optional[TokensEncoder]] = None
    """The embedding model for queries (if None, uses encoder)"""

    qlen: Param[int] = 20
    """Maximum query length (this can be even shortened by the model)"""

    dlen: Param[int] = 2000
    """Maximum document length (this can be even shortened by the model)"""

    def __initialize__(self, options):
        self.encoder.initialize(options)
        if self.query_encoder:
            self.encoder.initialize(options)
        else:
            self.query_encoder = self.encoder

    def __validate__(self):
        assert (
            self.dlen <= self.encoder.maxtokens()
        ), f"The maximum document length ({self.dlen}) should be less "
        "that what the vocab can process ({self.encoder.maxtokens()})"
        assert (
            self.qlen <= self.encoder.maxtokens()
        ), f"The maximum query length ({self.qlen}) should be less "
        "that what the vocab can process ({self.encoder.maxtokens()})"

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        return self._forward(inputs, info)

    def _forward(self, inputs: BaseRecords, info: TrainerContext = None):
        raise NotImplementedError
