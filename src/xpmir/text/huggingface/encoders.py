import sys
from typing import Optional
from experimaestro import Param
from xpmir.learning import Module
from xpmir.text.encoders import (
    TextsRepresentationOutput,
    TokenizedEncoder,
    TokenizedTexts,
    TokensRepresentationOutput,
)

from .base import HFModel


class HFEncoderBase(Module):
    """Base HuggingFace encoder"""

    model: Param[HFModel]
    """A Hugging-Face model"""

    @classmethod
    def from_pretrained_id(cls, model_id: str):
        """Returns a new encoder

        :param model_id: The HuggingFace Hub ID
        :return: A hugging-fasce based encoder
        """
        return cls(
            model=HFModel.from_pretrained_id(model_id),
        )

    def __initialize__(self, options):
        super().__initialize__(options)
        self.model.initialize(options)

    def static(self):
        """Embeddings from transformers are learnable"""
        return False

    def dimension(self):
        return self.model.hf_config.hidden_size

    @property
    def max_length(self):
        """Returns the maximum length that the model can process"""
        return sys.maxsize


class HFTokensEncoder(
    HFEncoderBase, TokenizedEncoder[TokenizedTexts, TokensRepresentationOutput]
):
    """HuggingFace-based tokenized"""

    def dim(self):
        return self.tokenizer.dimension

    def forward(self, tokenized: TokenizedTexts) -> TokensRepresentationOutput:
        tokenized = tokenized.to(self.model.contextual_model.device)
        y = self.model.contextual_model(
            tokenized.ids, attention_mask=tokenized.mask.to(self.device)
        )
        return TokensRepresentationOutput(
            tokenized=tokenized, value=y.last_hidden_state
        )


class HFCLSEncoder(
    HFEncoderBase, TokenizedEncoder[TokenizedTexts, TextsRepresentationOutput]
):
    """Encodes a text using the [CLS] token"""

    maxlen: Param[Optional[int]] = None
    """Limit the text to be encoded"""

    def forward(self, tokenized: TokenizedTexts) -> TextsRepresentationOutput:
        tokenized = tokenized.to(self.device)
        y = self.model.contextual_model(tokenized.ids, attention_mask=tokenized.mask)

        # Assumes that [CLS] is the first token
        return TextsRepresentationOutput(
            tokenized=tokenized, value=y.last_hidden_state[:, 0]
        )
