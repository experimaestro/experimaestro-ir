# ColBERT implementation
#
# From
# https://github.com/stanford-futuredata/ColBERT/blob/v0.2/colbert/modeling/colbert.py

from typing import List
from experimaestro import Config, Constant, Param, default, Annotated
import torch
from torch import nn
import torch.nn.functional as F
from xpmir.letor.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.neural.interaction import InteractionScorer


class Similarity(Config):
    def __call__(self, queries, documents) -> torch.Tensor:
        raise NotImplementedError()


class L2Distance(Similarity):
    def __call__(self, queries, documents):
        return (
            (-1.0 * ((queries.unsqueeze(2) - documents.unsqueeze(1)) ** 2).sum(-1))
            .max(-1)
            .values.sum(-1)
        )


class CosineDistance(Similarity):
    def __call__(self, queries, documents):
        return (queries @ documents.permute(0, 2, 1)).max(2).values.sum(1)


class Colbert(InteractionScorer):
    """ColBERT model

    Implementation of the Colbert model from:

        Khattab, Omar, and Matei Zaharia. “ColBERT: Efficient and Effective
        Passage Search via Contextualized Late Interaction over BERT.” SIGIR
        2020, Xi'An, China

    For the standard Colbert model, use BERT as the vocab(ulary)
    """

    version: Constant[int] = 2
    """Current version of the code (changes when a bug is found)"""

    masktoken: Param[bool] = True
    """Whether a [MASK] token should be used instead of padding"""

    querytoken: Param[bool] = True
    """Whether a specific query token should be used as a prefix to the question"""

    doctoken: Param[bool] = True
    """Whether a specific document token should be used as a prefix to the document"""

    similarity: Annotated[Similarity, default(CosineDistance())]
    """Which similarity to use"""

    linear_dim: Param[int] = 128
    """Size of the last linear layer (before computing inner products)"""

    compression_size: Param[int] = 128
    """Projection layer for the last layer (or 0 if None)"""

    def __validate__(self):
        super().__validate__()
        assert not self.vocab.static(), "The vocabulary should be learnable"

        assert self.compression_size >= 0, "Last layer size should be 0 or above"

        # TODO: implement the "official" Colbert
        assert not self.masktoken, "Not implemented"
        assert not self.querytoken, "Not implemented"
        assert not self.doctoken, "Not implemented"

    def _initialize(self, random):
        super()._initialize(random)

        self.linear = nn.Linear(self.vocab.dim(), self.linear_dim, bias=False)

    def _encode(self, texts: List[str], maskoutput=False):
        tokens = self.vocab.batch_tokenize(texts, mask=maskoutput)
        output = self.linear(self.vocab(tokens))

        if maskoutput:
            mask = tokens.mask.unsqueeze(2).float().to(output.device)
            output = output * mask

        return F.normalize(output, p=2, dim=2)

    def _forward(self, inputs: BaseRecords, info: TrainerContext = None):
        queries = self._encode([q.text for q in inputs.queries], False)
        documents = self._encode([d.text for d in inputs.documents], True)

        return self.similarity(queries, documents)
