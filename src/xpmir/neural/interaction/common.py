from abc import ABC, abstractmethod
from typing import Union

import torch
from attrs import define
from experimaestro import Config

from xpmir.learning.batchers import Sliceable


@define
class SimilarityInput(Sliceable["SimilarityInput"]):
    value: torch.Tensor
    """A 3D tensor (batch x max_length x dim)"""

    mask: torch.BoolTensor
    """Mask for tokens of shape (batch x max_length)"""

    def __getitem__(self, index: Union[int, slice]) -> "SimilarityInput":
        return SimilarityInput(self.value[index], self.mask[index])

    def __len__(self) -> int:
        return len(self.value)


@define
class SimilarityOutput:
    """Output for token similarities"""

    similarity: torch.Tensor
    """Similarity of each token

    The shape (Bq x Lq x Bd x Ld) when computing products, or (B x Lq x Ld) when
    computing pairs"""

    q_mask: torch.BoolTensor
    """Mask for query tokens (broadcastable)"""

    d_mask: torch.BoolTensor
    """Mask for document tokens (broadcastable)"""


class Similarity(Config, ABC):
    """Base class for similarity between two texts representations (3D tensor
    batch x length x dim)"""

    def preprocess(self, encoded: SimilarityInput) -> SimilarityInput:
        """Optional preprocessing"""
        return encoded

    @abstractmethod
    def compute_pairs(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
    ) -> SimilarityOutput:
        """Computes the score of queries and documents

        :param queries: Queries as a (B1 x L1 x dim) tensor
        :param documents: Documents as a (B2 x L2 x dim) tensor with B2 = B1 if
            product is False
        :param product: Computes the scores between all queries and documents
        :return: The score of all topics/documents (B1 x B2)
        """
        ...

    @abstractmethod
    def compute_product(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
    ) -> SimilarityOutput:
        """Computes the score of queries and documents

        :param queries: Queries as a (B x L1 x dim) tensor
        :param documents: Documents as a (B x L2 x dim) tensor with B2 = B1 if
            product is False
        :param product: Computes the scores between all queries and documents
        :return: The similarity of query/document pairs (B)
        """
        ...


class DotProductSimilarity(Similarity):
    def compute_product(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
    ):
        q, q_mask = queries.value, queries.mask
        d, d_mask = documents.value, documents.mask

        # Bq x Lq x Bd x Ld
        inner = (q.flatten(0, 1) @ d.flatten(0, 1).transpose(0, 1)).reshape(
            q.shape[:2] + d.shape[:2]
        )

        # Reshape query of document masks
        d_mask = d_mask.reshape(1, 1, *d_mask.shape)
        q_mask = q_mask.reshape(*q_mask.shape, 1, 1)

        # Max on document tokens, sum over query tokens
        return SimilarityOutput(inner, q_mask, d_mask)

    def compute_pairs(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
    ):
        q, q_mask = queries.value, queries.mask
        d, d_mask = documents.value, documents.mask

        # B x Lq x Ld
        assert len(q) == len(d), "Batch sizes don't match"
        inner = q @ d.transpose(1, 2)

        # Reshape query of document masks
        q_mask = q_mask.unsqueeze(2)
        d_mask = d_mask.unsqueeze(1)

        return SimilarityOutput(inner, q_mask, d_mask)


class CosineSimilarity(DotProductSimilarity):
    """Cosine similarity between two texts representations (3D tensor batch x
    length x dim)"""

    def preprocess(self, output: SimilarityInput):
        value = output.value / output.value.norm(p="fro", dim=-1, keepdim=True)
        return SimilarityInput(value, output.mask)
