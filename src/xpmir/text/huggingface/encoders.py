import torch
from typing import List, Optional
from experimaestro import copyconfig, Param
from xpmir.text.encoders import (
    TextEncoder,
    DualTextEncoder,
    TripletTextEncoder,
    ListTextEncoder,
)
from .base import HFBaseModel
from .tokenizers import HFTokenizerBase, HFTokenizer, HFListTokenizer


class HFTextEncoder(TextEncoder, DualTextEncoder, TripletTextEncoder):
    """Encodes a text using the [CLS] token"""

    tokenizer: Param[HFTokenizerBase]
    model: Param[HFBaseModel]

    maxlen: Param[Optional[int]] = None
    """Limit the text to be encoded"""

    def forward(self, texts: List) -> torch.Tensor:
        tokenized = self.tokenizer.batch_tokenize(texts, maxlen=self.maxlen, mask=True)

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


class HFTextEncoder(HFTextEncoder, TextEncoder, DualTextEncoder):
    tokenizer: Param[HFTokenizer]


class HFListTextEncoder(HFTextEncoder, ListTextEncoder):
    tokenizer: Param[HFListTokenizer]
