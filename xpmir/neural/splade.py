from typing import List
from experimaestro import Config, Param
import torch.nn as nn
import torch
from xpmir.neural import TorchLearnableScorer
from xpmir.vocab.huggingface import TransformerVocab
from xpmir.letor.records import BaseRecords


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

    def __cal__(self, logits, mask):
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

    encoder: Param[TransformerVocab]
    aggregation: Param[Aggregation]

    def initialize(self, random):
        super().initialize(random)

        self.encoder.initialize()

    def forward(self, inputs: BaseRecords):
        queries = self._encode([q.text for q in inputs.queries])
        documents = self._encode([d.text for d in inputs.documents])

    def _encode(self, texts: List[str]):
        tokenized = self.vocab.batch_tokenize(inputs, maskoutput=True)
        out = self.vocab(input_ids=tokenized.ids, attention_mask=tokenized.mask)
        return self.aggregation(out)
