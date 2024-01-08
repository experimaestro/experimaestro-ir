from typing import List, Optional

from experimaestro import Config, Param
from xpmir.text.encoders import (
    TextsRepresentationOutput,
    TokenizedEncoder,
    TokenizedTexts,
    TokensRepresentationOutput,
)

from .base import HFModel
from .tokenizers import HFTokenizer, HFTokenizerBase


class HFEncoderBase(Config):
    """Base HuggingFace encoder"""

    tokenizer: Param[HFTokenizerBase]
    """The tokenizer"""

    model: Param[HFModel]
    """A Hugging-Face model"""

    @classmethod
    def from_pretrained_id(cls, model_id: str):
        """Returns a new encoder

        :param model_id: The HuggingFace Hub ID
        :return: A hugging-fasce based encoder
        """
        return cls(
            tokenizer=HFTokenizer(model_id=model_id),
            model=HFModel.from_pretrained_id(model_id),
        )


class HFTokensEncoder(
    HFEncoderBase, TokenizedEncoder[TokenizedTexts, TokensRepresentationOutput]
):
    """HuggingFace-based tokenized"""

    def dim(self):
        return self.tokenizer.dimension

    def forward(self, tokenized: TokenizedTexts) -> TokensRepresentationOutput:
        y = self.model.contextual_model(
            tokenized.ids, attention_mask=s.mask.to(self.device)
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
        y = self.model.contextual_model(
            tokenized.ids, attention_mask=tokenized.mask.to(self.device)
        )

        # Assumes that [CLS] is the first token
        return TextsRepresentationOutput(
            tokenized=tokenized, value=y.last_hidden_state[:, 0]
        )
