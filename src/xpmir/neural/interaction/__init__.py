import torch
from experimaestro import Param
from xpmir.rankers import LearnableScorer
from xpmir.text import TokensEncoder
from xpmir.letor.records import BaseRecords
from xpmir.learning.context import TrainerContext


class InteractionScorer(LearnableScorer):
    """Interaction-based neural scorer

    This is the base class for all scorers that depend on a map
    of cosine/inner products between query and document token representations.

    Attributes:

        vocab: The embedding model -- the vocab also defines how to tokenize text
        qlen: Maximum query length (this can be even shortened by the model)
        dlen: Maximum document length (this can be even shortened by the model)
    """

    vocab: Param[TokensEncoder]
    qlen: Param[int] = 20
    dlen: Param[int] = 2000

    def _initialize(self, random):
        self.random = random
        self.vocab.initialize()

    def __validate__(self):
        assert (
            self.dlen <= self.vocab.maxtokens()
        ), f"The maximum document length ({self.dlen}) should be less "
        "that what the vocab can process ({self.vocab.maxtokens()})"
        assert (
            self.qlen <= self.vocab.maxtokens()
        ), f"The maximum query length ({self.qlen}) should be less "
        "that what the vocab can process ({self.vocab.maxtokens()})"

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        return self._forward(inputs, info)

    def _forward(self, inputs: BaseRecords, info: TrainerContext = None):
        raise NotImplementedError

    def save(self, path):
        state = self.state_dict(keep_vars=True)
        for key in list(state):
            if state[key].requires_grad:
                state[key] = state[key].data
            else:
                del state[key]
        torch.save(state, path)

    def load(self, path):
        self.load_state_dict(torch.load(path), strict=False)
