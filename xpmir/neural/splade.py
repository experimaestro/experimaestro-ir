from experimaestro import Config, Param
import torch.nn as nn
from xpmir.neural import TorchLearnableScorer
from xpmir.vocab.huggingface import TransformerVocab


class Aggregation(Config):
    """The aggregation function for Splade"""

    pass


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

    def forward(self):
        out = self.transformer(**kwargs)[
            "logits"
        ]  # output (logits) of MLM head, shape (bs, pad_len, voc_size)
        if self.agg == "max":
            values, _ = torch.max(
                torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1),
                dim=1,
            )
            return values
            # 0 masking also works with max because all activations are positive
        else:
            return torch.sum(
                torch.log(1 + torch.relu(out)) * kwargs["attention_mask"].unsqueeze(-1),
                dim=1,
            )
