import torch
from typing import List
from experimaestro import Param
from .encoders import TextEncoder, TokensEncoder


class MeanTextEncoder(TextEncoder):
    """Returns the mean of the word embeddings"""

    encoder: Param[TokensEncoder]

    def __initialize__(self, options):
        self.encoder.__initialize__(options)

    def static(self):
        return self.encoder.static()

    @property
    def dimension(self):
        return self.encoder.dim()

    def forward(self, texts: List[str]) -> torch.Tensor:
        tokenized = self.encoder.batch_tokenize(texts, True)
        emb_texts = self.encoder(tokenized)
        # Computes the mean over the time dimension (vocab output is batch x time x dim)
        return emb_texts.mean(1)
