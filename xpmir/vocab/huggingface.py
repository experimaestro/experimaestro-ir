from experimaestro.compat import cached_property
from typing import List, Optional, Tuple, Union
import logging
import torch
import torch.nn as nn
from experimaestro import Param
from xpmir.vocab.encoders import DualTextEncoder, TextEncoder

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise

from xpmir.letor.records import TokenizedTexts
import xpmir.vocab as vocab


class TransformerVocab(vocab.Vocab):
    """Transformer-based encoder

    Args:

    model_id: Model ID from huggingface
    trainable: Whether BERT parameters should be trained
    layer: Layer to use (0 is the last, -1 to use them all)
    """

    model_id: Param[str] = "bert-base-uncased"
    trainable: Param[bool] = False
    layer: Param[int] = 0

    CLS: int
    SEP: int

    @cached_property
    def tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

    @property
    def pad_tokenid(self) -> int:
        return self.tokenizer.pad_token_id

    def initialize(self, noinit=False):
        super().initialize(noinit=noinit)

        if noinit:
            config = AutoConfig.from_pretrained(self.model_id)
            self.model = AutoModel.from_config(config)
        else:
            self.model = AutoModel.from_pretrained(self.model_id)

        # Loads the tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)

        layer = self.layer
        if layer == -1:
            layer = None
        self.CLS = self.tokenizer.cls_token_id
        self.SEP = self.tokenizer.sep_token_id

        if self.trainable:
            self.model.train()
        else:
            self.model.eval()

    def train(self, mode: bool = True):
        # We should not make this layer trainable unless asked
        if mode:
            if self.trainable:
                self.model.train(mode)
        else:
            self.model.train(mode)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def tok2id(self, tok):
        return self.tokenizer.vocab[tok]

    def static(self):
        return not self.trainable

    def batch_tokenize(
        self,
        texts: Union[List[str], List[Tuple[str, str]]],
        batch_first=True,
        maxlen=None,
        mask=False,
    ) -> TokenizedTexts:
        if maxlen is None:
            maxlen = self.tokenizer.model_max_length
        else:
            maxlen = min(maxlen, self.tokenizer.model_max_length)

        assert batch_first, "Batch first is the only option"

        r = self.tokenizer(
            list(texts),
            max_length=maxlen,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_length=True,
            return_attention_mask=mask,
        )
        return TokenizedTexts(
            None,
            r["input_ids"].to(self.device),
            r["length"],
            r.get("attention_mask", None),
        )

    def id2tok(self, idx):
        if torch.is_tensor(idx):
            if len(idx.shape) == 0:
                return self.id2tok(idx.item())
            return [self.id2tok(x) for x in idx]
        # return self.tokenizer.ids_to_tokens[idx]
        return self.tokenizer.id_to_token(idx)

    def lexicon_size(self) -> int:
        return self.tokenizer._tokenizer.get_vocab_size()

    def maxtokens(self) -> int:
        return self.tokenizer.model_max_length

    def forward(self, toks: TokenizedTexts):
        device = self._dummy_params.device
        return self.model(
            toks.ids.to(device),
            attention_mask=toks.mask.to(device) if toks.mask is not None else None,
        ).last_hidden_state

    def dim(self):
        return self.model.config.hidden_size


class IndependentTransformerVocab(TransformerVocab):
    """Encodes as [CLS] QUERY [SEP]"""

    def __call__(self, tokids):
        with torch.set_grad_enabled(self.trainable):
            y = self.model(tokids)

        return y.last_hidden_state


class TransformerEncoder(TransformerVocab, TextEncoder):
    """Encodes using the [CLS] token"""

    def forward(self, texts: List[str]):
        tokenized = self.batch_tokenize(texts)

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            y = self.model(tokenized.ids)

        return y.last_hidden_state[:, -1]


class DualTransformerEncoder(TransformerVocab, DualTextEncoder):
    """Encodes the (query, document pair) using the [CLS] token

    maxlen: Maximum length of the query document pair (in tokens) or None if using the transformer limit
    """

    maxlen: Param[Optional[int]] = None

    def forward(self, texts: List[Tuple[str, str]]):
        device = self._dummy_params.device
        tokenized = self.batch_tokenize(texts, maxlen=self.maxlen, mask=True)

        with torch.set_grad_enabled(torch.is_grad_enabled() and self.trainable):
            y = self.model(tokenized.ids, attention_mask=tokenized.mask.to(device))

        return y.last_hidden_state[:, -1]

    @property
    def dimension(self) -> int:
        return self.model.config.hidden_size
