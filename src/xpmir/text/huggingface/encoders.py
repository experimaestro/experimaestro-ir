import torch
from typing import List, Optional, Union, Tuple
from experimaestro import copyconfig, Param
from xpmir.text.encoders import TextEncoder, DualTextEncoder, TokensEncoder
from .base import HFTokenizer, HFBaseModel


class HFTextEncoder(TextEncoder, DualTextEncoder):
    """Encodes a text (or a pair of texts) using the [CLS] token"""

    tokenizer: Param[HFTokenizer]
    model: Param[HFBaseModel]

    maxlen: Param[Optional[int]] = None
    """Limit the text to be encoded"""

    def forward(
        self, texts: Union[List[str], List[Tuple[str, str]]], maxlen=None
    ) -> torch.Tensor:
        tokenized = self.tokenizer.batch_tokenize(
            texts, maxlen=maxlen or self.maxlen, mask=True
        )

        y = self.model(tokenized.ids, attention_mask=tokenized.mask.to(self.device))

        # Assumes that [CLS] is the first token
        return y.last_hidden_state[:, 0]

    @property
    def dimension(self):
        return self.dim()

    def with_maxlength(self, maxlen: int):
        return copyconfig(self, maxlen=maxlen)

    def distribute_models(self, update):
        self.model = update(self.model)


class HFTextListTokensEncoder(TokensEncoder):
    """Encodes a text (composed of a list of texts) using the [CLS] token"""

    tokenizer: Param[HFTokenizer]
    model: Param[HFBaseModel]

    maxlen: Param[Optional[int]] = None
    """Limit the text to be encoded"""

    def forward(
        self, texts: Union[List[str], List[Tuple[str, str]]], maxlen=None
    ) -> torch.Tensor:
        tokenized = self.tokenizer.batch_tokenize(
            texts, maxlen=maxlen or self.maxlen, mask=True
        )

        y = self.model(tokenized.ids, attention_mask=tokenized.mask.to(self.device))

        # Assumes that [CLS] is the first token
        return y.last_hidden_state[:, 0]

    @property
    def dimension(self):
        return self.dim()

    def with_maxlength(self, maxlen: int):
        return copyconfig(self, maxlen=maxlen)

    def distribute_models(self, update):
        self.model = update(self.model)
