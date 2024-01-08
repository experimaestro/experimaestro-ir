import torch
from abc import ABC, abstractmethod
from experimaestro import Config


class Similarity(Config, ABC):
    """Base class for similarity between two texts representations (3D tensor batch x length x dim)"""

    @abstractmethod
    def __call__(self, queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        ...


class L2Distance(Similarity):
    def __call__(self, queries: torch.Tensor, documents: torch.Tensor):
        return (
            (-1.0 * ((queries.unsqueeze(2) - documents.unsqueeze(1)) ** 2).sum(-1))
            .max(-1)
            .values.sum(-1)
        )


class CosineSimilarity(Similarity):
    """Cosine similarity between two texts representations (3D tensor batch x length x dim)"""

    def __call__(self, queries: torch.Tensor, documents: torch.Tensor):
        return (queries @ documents.permute(0, 2, 1)).max(2).values.sum(1)
