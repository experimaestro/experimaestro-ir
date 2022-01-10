from typing import Generic, Iterable, List, Optional, TypeVar
import torch
import torch.nn as nn

from experimaestro import Param
from xpmir.letor.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.letor import Module
from xpmir.rankers import LearnableScorer


class TorchLearnableScorer(LearnableScorer, Module):
    """Base class for torch-learnable scorers"""

    def __init__(self):
        nn.Module.__init__(self)
        super().__init__()

    def __call__(self, inputs: BaseRecords, info: TrainerContext = None):
        # Redirects to nn.Module rather than using LearnableScorer one
        return nn.Module.__call__(self, inputs, info)

    def train(self, mode=True):
        return nn.Module.train(self, mode)


class DualRepresentationScorer(TorchLearnableScorer):
    """Neural scorer based on (at least a partially) independant representation
    of

    This is the base class for all scorers that depend on a map
    of cosine/inner products between query and document tokens.
    """

    def forward(self, inputs: BaseRecords, info: Optional[TrainerContext] = None):
        # Forward to model
        enc_queries = self.encode_queries([q.text for q in inputs.unique_queries])
        enc_documents = self.encode_documents([d.text for d in inputs.unique_documents])

        # Get the pairs
        pairs = inputs.pairs()
        q_ix, d_ix = pairs

        # TODO: Use a product query x document if possible

        return self.score_pairs(
            enc_queries[
                q_ix,
            ],
            enc_documents[
                d_ix,
            ],
            info,
        )

    def encode(self, texts: Iterable[str]):
        raise NotImplementedError()

    def encode_documents(self, texts: Iterable[str]):
        return self.encode(texts)

    def encode_queries(self, texts: Iterable[str]):
        return self.encode(texts)

    def score_product(self, queries, documents, info: TrainerContext):
        raise NotImplementedError()

    def score_pairs(
        self, queries, documents, info: Optional[TrainerContext]
    ) -> torch.Tensor:
        raise NotImplementedError()
