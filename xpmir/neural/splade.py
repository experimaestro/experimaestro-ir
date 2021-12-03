from typing import List, Tuple
from experimaestro import Config, Param
import torch.nn as nn
import torch
from xpmir.neural import TorchLearnableScorer
from xpmir.vocab.huggingface import TransformerVocab
from xpmir.letor.records import BaseRecords
from xpmir.letor.trainers.batchwise import BatchwiseTrainer
from transformers import AutoModelForMaskedLM


class Aggregation(Config):
    """The aggregation function for Splade"""

    pass


class MaxAggregation(Aggregation):
    """Aggregate using a max"""

    def __call__(self, logits, mask):
        values, _ = torch.max(
            torch.log(1 + torch.relu(logits) * mask.unsqueeze(-1)),
            dim=1,
        )
        return values


class SumAggregation(Aggregation):
    """Aggregate using a sum"""

    def __call__(self, logits, mask):
        return torch.sum(
            torch.log(1 + torch.relu(logits) * mask.unsqueeze(-1)),
            dim=1,
        )


class Splade(TorchLearnableScorer):
    """Splade model

        SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval (arXiv:2109.10086)

    Attributes:

        encoder: The transformer that will be fine-tuned
    """

    AUTOMODEL_CLASS: AutoModelForMaskedLM
    encoder: Param[TransformerVocab]
    aggregation: Param[Aggregation]

    def initialize(self, random):
        super().initialize(random)

        self.encoder.initialize()

    def score_pairs(self, queries: torch.Tensor, documents: torch.Tensor):
        return queries.unsqueeze(1) @ documents.unsqueeze(2).flatten()

    def score_product(self, queries: torch.Tensor, documents: torch.Tensor):
        return (queries @ documents.T).flatten()

    def encode(self, texts: List[str]) -> torch.Tensor:
        """Returns a batch x vocab tensor"""
        tokenized = self.vocab.batch_tokenize(inputs, maskoutput=True)
        out = self.vocab(input_ids=tokenized.ids, attention_mask=tokenized.mask)
        return self.aggregation(out)


def spladev2(sampler) -> Tuple[BatchwiseTrainer, Splade]:
    """Returns the model described in Splade V2"""
    from xpmir.letor.optim import Adam

    optimizer = Adam(lr=2e-5)
    BatchwiseTrainer(sampler=sampler, lossfn=...)
