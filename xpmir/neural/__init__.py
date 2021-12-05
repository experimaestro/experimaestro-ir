from typing import Generic, Iterable, List, Optional, TypeVar
import torch
import torch.nn as nn

from experimaestro import Param
from xpmir.letor.traininfo import TrainingInformation
from xpmir.letor.records import BaseRecords, Document, Query
from xpmir.rankers import LearnableScorer
from xpmir.vocab import Vocab


class TorchLearnableScorer(LearnableScorer, nn.Module):
    """Base class for torch-learnable scorers"""

    def __init__(self):
        nn.Module.__init__(self)

    def __call__(self, inputs: BaseRecords, info: TrainingInformation = None):
        return nn.Module.__call__(self, inputs, metrics)


class SeparateRepresentationTorchScorer(TorchLearnableScorer):
    """Neural scorer based on (at least a partially) independant representation
    of

    This is the base class for all scorers that depend on a map
    of cosine/inner products between query and document tokens.
    """

    def forward(self, inputs: BaseRecords, info: TrainingInformation = None):
        # Forward to model
        enc_queries = [self._encode_queries(q.text) for q in inputs.unique_queries]
        enc_documents = [self._encode_document(d.text) for d in inputs.unique_documents]

        pairs = inputs.pairs
        if pairs is None:
            self.score_product(enc_queries, enc_documents)

        # Case where pairs of indices are given
        q_ix, d_ix = pairs
        device = enc_queries.device

        return self.score_pairs(
            torch.index_select(enc_queries, 0, torch.LongTensor(q_ix, device=device)),
            torch.index_select(enc_documents, 0, torch.LongTensor(d_ix, device=device)),
        )

    def encode(self, texts: Iterable[str]):
        raise NotImplementedError()

    def encode_documents(self, texts: Iterable[str]):
        return self.encode(texts)

    def encode_queries(self, texts: Iterable[str]):
        return self.encode(texts)


class InteractionScorer(TorchLearnableScorer):
    """Interaction-based neural scorer

    This is the base class for all scorers that depend on a map
    of cosine/inner products between query and document tokens.

    Attributes:

        vocab: The embedding model -- the vocab also defines how to tokenize text
        qlen: Maximum query length (this can be even shortened by the model)
        dlen: Maximum document length (this can be even shortened by the model)
        add_runscore:
            Whether the base predictor score should be added to the
            model score
    """

    vocab: Param[Vocab]
    qlen: Param[int] = 20
    dlen: Param[int] = 2000

    def initialize(self, random):
        self.random = random
        self.vocab.initialize()

    def __validate__(self):
        assert (
            self.dlen <= self.vocab.maxtokens()
        ), f"The maximum document length ({self.dlen}) should be less that what the vocab can process ({self.vocab.maxtokens()})"
        assert (
            self.qlen <= self.vocab.maxtokens()
        ), f"The maximum query length ({self.qlen}) should be less that what the vocab can process ({self.vocab.maxtokens()})"

    def forward(self, inputs: BaseRecords, info: TrainingInformation = None):
        # Forward to model
        result = self._forward(inputs, metrics)

        return result

    def _forward(self, inputs: BaseRecords, info: TrainingInformation = None):
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
