from functools import cached_property
from typing import List
import torch

from experimaestro import config, Param

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise

from xpmir.letor.samplers import TokenizedTexts
import xpmir.vocab as vocab


@config()
class TransformerVocab(vocab.Vocab):
    """
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
        raise self.tokenizer.pad_token_id

    def initialize(self):
        super().initialize()
        self.model = AutoModel.from_pretrained(self.model_id)
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
        self, texts: List[str], batch_first=True, maxlen=None
    ) -> TokenizedTexts:
        maxlen = max(maxlen, self.tokenizer.model_max_length)
        r = self.tokenizer(
            list(texts),
            max_length=maxlen,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_length=True,
        )
        return TokenizedTexts(None, r["input_ids"].to(self.device), r["length"])

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

    def forward(self, toks, lens=None):
        return self.model(toks).last_hidden_state


@config()
class IndependentTransformerVocab(TransformerVocab):
    """Encodes as [CLS] QUERY [SEP]"""

    def __call__(self, tokids):
        with torch.set_grad_enabled(self.trainable):
            y = self.model(tokids)

        return y.last_hidden_state
