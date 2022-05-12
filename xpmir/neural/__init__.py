from typing import Generic, Iterable, List, Optional, TypeVar
import torch
import torch.nn as nn

from xpmir.letor.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.letor.optim import Module
from xpmir.rankers import LearnableScorer


class TorchLearnableScorer(LearnableScorer, Module):
    """Base class for torch-learnable scorers"""

    def __init__(self):
        nn.Module.__init__(self)
        super().__init__()

    __call__ = nn.Module.__call__
    to = nn.Module.to

    def train(self, mode=True):
        return nn.Module.train(self, mode)


class DualRepresentationScorer(TorchLearnableScorer):
    """Neural scorer based on (at least a partially) independant representation
    of the document and the question.

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
        """Encode a list of texts (document or query)

        The return value is model dependent"""
        raise NotImplementedError()

    def encode_documents(self, texts: Iterable[str]):
        """Encode a list of texts (document or query)

        The return value is model dependent"""
        return self.encode(texts)

    def encode_queries(self, texts: Iterable[str]):
        """Encode a list of texts (document or query)

        The return value is model dependent"""
        return self.encode(texts)

    def score_product(self, queries, documents, info: Optional[TrainerContext]):
        """Computes the score of all possible pairs of query and document

        Args:
            queries (Any): The encoded queries
            documents (Any): The encoded documents
            info (Optional[TrainerContext]): The training context (if learning)

        Returns:
            torch.Tensor:
                A tensor of dimension (N, P) where N is the number of queries
                and P the number of documents
        """
        raise NotImplementedError()

    def score_pairs(
        self, queries, documents, info: Optional[TrainerContext]
    ) -> torch.Tensor:
        """Score the specified pairs of queries/documents.

        There are as many queries as documents. The exact type of
        queries and documents depends on the specific instance of the
        dual representation scorer.

        Args:
            queries (Any): The list of encoded queries
            documents (Any): The matching list of encoded documents
            info (Optional[TrainerContext]): _description_

        Returns:
            torch.Tensor:
                A tensor of dimension (N, 2) where N is the number of documents/queries
        """
        raise NotImplementedError()
