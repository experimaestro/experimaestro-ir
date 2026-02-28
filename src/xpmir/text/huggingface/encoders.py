import re
import sys
import logging
from typing import List, Optional
from functools import cached_property

import torch
import torch.nn as nn
from experimaestro import Param, Constant
from xpm_torch import Module
from xpm_torch.learner import TrainState
from xpm_torch.parameters import ParametersIterator
from xpmir.text.encoders import (
    TextEncoder,
    TextsRepresentationOutput,
    TokenizedEncoder,
    TokenizedTexts,
    TokensRepresentationOutput,
)

from .base import HFModel

try:
    from transformers import AutoTokenizer
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise

logger = logging.getLogger(__name__)


class HFEncoderBase(Module):
    """Base HuggingFace encoder"""

    model: Param[HFModel]
    """A Hugging-Face model"""

    @classmethod
    def from_pretrained_id(cls, model_id: str, **kwargs):
        """Returns a new encoder

        :param model_id: The HuggingFace Hub ID
        :param kwargs: keyword arguments passed to the model constructor
        :return: A hugging-fasce based encoder
        """
        return cls(model=HFModel.from_pretrained_id(model_id), **kwargs)

    def __initialize__(self):
        super().__initialize__()
        self.model.initialize()

    def static(self):
        """Embeddings from transformers are learnable"""
        return False

    @property
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
            tokenized.ids,
            attention_mask=tokenized.mask.to(self.device),
            token_type_ids=tokenized.token_type_ids,
        )
        return TokensRepresentationOutput(
            tokenized=tokenized, value=y.last_hidden_state
        )


class HFCLSEncoder(
    HFEncoderBase, TokenizedEncoder[TokenizedTexts, TextsRepresentationOutput]
):
    """Encodes a text using the [CLS] token"""

    def forward(self, tokenized: TokenizedTexts) -> TextsRepresentationOutput:
        tokenized = tokenized.to(self.device)
        y = self.model.contextual_model(
            tokenized.ids,
            attention_mask=tokenized.mask,
            token_type_ids=tokenized.token_type_ids,
        )

        # Assumes that [CLS] is the first token
        return TextsRepresentationOutput(
            tokenized=tokenized, value=y.last_hidden_state[:, 0]
        )


class SentenceTransformerTextEncoder(TextEncoder):
    """A Sentence Transformers text encoder"""

    model_id: Param[str] = "sentence-transformers/all-MiniLM-L6-v2"

    def __initialize__(self):
        super().__initialize__()
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.model_id)

    def forward(self, texts: List[str]) -> torch.Tensor:
        return self.model.encode(texts)


class OneHotHuggingFaceEncoder(TextEncoder):
    """A tokenizer which encodes the tokens into 0 and 1 vector
    1 represents the text contains the token and 0 otherwise"""

    model_id: Param[str] = "bert-base-uncased"
    """Model ID from huggingface"""

    maxlen: Param[Optional[int]] = None
    """Max length for texts"""

    version: Constant[int] = 2

    def __initialize__(self):
        super().__initialize__()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.CLS = self._tokenizer.cls_token_id
        self.SEP = self._tokenizer.sep_token_id
        self.PAD = self._tokenizer.pad_token_id
        self._dummy_params = nn.Parameter(torch.Tensor())

    @property
    def device(self):
        return self._dummy_params.device

    @cached_property
    def tokenizer(self):
        return self._tokenizer

    def batch_tokenize(self, texts):
        r = self.tokenizer(
            list(texts),
            max_length=self.maxlen,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        return r["input_ids"]

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a batch x vocab tensor"""
        tokenized_ids = self.batch_tokenize(texts)
        batch_size = len(texts)
        x = torch.zeros(batch_size, self.dimension)
        x[torch.arange(batch_size).unsqueeze(-1), tokenized_ids] = 1
        x[:, [self.PAD, self.SEP, self.CLS]] = 0
        return x.to(self.device)

    @property
    def dimension(self):
        return self.tokenizer.vocab_size

    def static(self):
        return False


class LayerSelector(ParametersIterator):
    """This class can be used to pick some of the transformer layers"""

    # For freezing everything except the embeddings
    re_layer: Param[str] = r"""(?:encoder|transformer)\.layer\.(\d+)\."""

    transformer: Param[HFModel]
    """The model for which layers are selected"""

    pick_layers: Param[int] = 0
    """Counting from the first processing layers (can be negative, i.e. -1 meaning
    until the last layer excluded, etc. / 0 means no layer)"""

    select_embeddings: Param[bool] = False
    """Whether to pick the embeddings layer"""

    select_feed_forward: Param[bool] = False
    """Whether to pick the feed forward of Transformer layers"""

    def __post_init__(self):
        self._re_layer = re.compile(self.re_layer)

    def __validate__(self):
        if (
            not (self.select_embeddings or self.select_feed_forward)
            and self.pick_layers == 0
        ):
            raise AssertionError("The layer selector will select nothing")

    @cached_property
    def nlayers(self):
        count = 0
        for name, _ in self.transformer.model.named_parameters():
            if m := self._re_layer.search(name):
                count = max(count, int(m.group(1)))
        return count

    def should_pick(self, name: str) -> bool:
        if self.select_embeddings and ("embeddings." in name):
            return True

        if self.select_feed_forward and ("intermediate" in name):
            return True

        if self.pick_layers != 0:
            if m := self._re_layer.search(name):
                layer = int(m.group(1))
                if self.pick_layers < 0:
                    return layer <= self.nlayers + self.pick_layers
                return layer < self.pick_layers

        return False

    def iter(self):
        for name, params in self.transformer.model.named_parameters():
            yield f"model.{name}", params, self.should_pick(name)

    def after(self, state: TrainState):
        if not self._initialized:
            self._initialized = True
            for name, param in self.transformer.model.named_parameters():
                if self.should_freeze(name):
                    logger.info("Freezing layer %s", name)
                    param.requires_grad = False
