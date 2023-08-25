import torch
from experimaestro import Config


class Similarity(Config):
    """A similarity between vector representations"""

    def __call__(self, queries: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class L2Distance(Similarity):
    def __call__(self, queries, documents):
        return (
            (-1.0 * ((queries.unsqueeze(2) - documents.unsqueeze(1)) ** 2).sum(-1))
            .max(-1)
            .values.sum(-1)
        )


class CosineSimilarity(Similarity):
    def __call__(self, queries, documents):
        return (queries @ documents.permute(0, 2, 1)).max(2).values.sum(1)
