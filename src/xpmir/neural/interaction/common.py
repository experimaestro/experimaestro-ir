from abc import ABC, abstractmethod
from typing import List, Union, Sequence
from attrs import evolve

import torch
from attrs import define
from experimaestro import Config


@define
class SimilarityInput(Sequence["SimilarityInput"]):
    value: torch.Tensor
    """A 3D tensor (batch x max_length x dim)"""

    mask: torch.BoolTensor
    """Mask for tokens of shape (batch x max_length)"""

    def __getitem__(self, index: Union[int, slice]) -> "SimilarityInput":
        return SimilarityInput(self.value[index], self.mask[index])

    def __len__(self) -> int:
        return len(self.value)


@define
class SimilarityInputWithTokens(SimilarityInput):
    tokens: List[List[str]]
    """A 3D tensor (batch x max_length x dim)"""

    def __getitem__(self, index: Union[int, slice]) -> "SimilarityInput":
        return SimilarityInputWithTokens(
            self.value[index], self.mask[index], self.tokens[index]
        )


@define
class SimilarityOutput(ABC):
    """Output for token similarities"""

    similarity: torch.Tensor
    """Similarity of each token

    The shape (Bq x Lq x Bd x Ld) when computing products, or (B x Lq x Ld) when
    computing pairs"""

    @abstractmethod
    def q_view(self, x: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def d_view(self, x: torch.Tensor) -> torch.Tensor:
        ...


@define
class PairsSimilarityOutput(SimilarityOutput):
    def q_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(2)

    def d_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(1)


@define
class ProductSimilarityOutput(SimilarityOutput):
    def d_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(1, 1, *x.shape)

    def q_view(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(*x.shape, 1, 1)


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
        q = queries.value
        d = documents.value

        # Bq x Lq x Bd x Ld
        inner = (q.flatten(0, 1) @ d.flatten(0, 1).transpose(0, 1)).reshape(
            q.shape[:2] + d.shape[:2]
        )

        # Max on document tokens, sum over query tokens
        return ProductSimilarityOutput(inner)

    def compute_pairs(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
    ):
        q = queries.value
        d = documents.value

        # B x Lq x Ld
        assert len(q) == len(d), "Batch sizes don't match"
        inner = q @ d.transpose(1, 2)

        return PairsSimilarityOutput(inner)


class CosineSimilarity(DotProductSimilarity):
    """Cosine similarity between two texts representations (3D tensor batch x
    length x dim)"""

    def preprocess(self, output: SimilarityInput):
        value = output.value / output.value.norm(p="fro", dim=-1, keepdim=True)
        return evolve(output, value=value)
