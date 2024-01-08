import logging
from typing import List, Tuple, Union

import torch
from experimaestro import Config, Param

from xpmir.learning.optim import ModuleInitOptions
from xpmir.text.tokenizers import (
    TokenizerBase,
    TokenizedTexts,
    TokenizerInput,
)

try:
    from transformers import AutoTokenizer
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise


HFTokenizerInput = Union[List[str], List[Tuple[str, str]]]


class HFTokenizer(Config):
    """This is the main tokenizer class"""

    model_id: Param[str]
    """The tokenizer hugginface ID"""

    def __initialize__(self, options: ModuleInitOptions):
        """Initialize the HuggingFace transformer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.cls = self.tokenizer.cls_token
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id

    def tokenize(
        self,
        texts: HFTokenizerInput,
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
            r.get("token_type_ids", None),
        )

    def id2tok(self, idx):
        """Returns the token strings corresponding to the token ids"""
        if torch.is_tensor(idx):
            if len(idx.shape) == 0:
                return self.id2tok(idx.item())
            return [self.id2tok(x) for x in idx]
        return self.tokenizer.id_to_token(idx)

    def lexicon_size(self) -> int:
        return self.tokenizer._tokenizer.get_vocab_size()

    def maxtokens(self) -> int:
        return self.tokenizer.model_max_length

    def dim(self):
        return self.tokenizer.config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary"""
        return self.tokenizer.vocab_size


class HFTokenizerBase(TokenizerBase[TokenizerInput, TokenizedTexts]):
    """Base class for all Hugging-Face tokenizers"""

    tokenizer: Param[HFTokenizer]
    """The HuggingFace tokenizer"""

    @classmethod
    def from_pretrained_id(cls, hf_id: str):
        return cls(tokenizer=HFTokenizer(model_id=hf_id))

    def vocabulary_size(self):
        return self.tokenizer.vocab_size

    def tok2id(self, tok: str) -> int:
        return self.tokenizer.tok2id(tok)

    def id2tok(self, idx: int) -> str:
        return self.tokenizer.id2tok(idx)


class HFStringTokenizer(HFTokenizerBase[HFTokenizerInput]):
    """Process list of texts"""

    def tokenize(self, texts: List[str]) -> TokenizedTexts:
        assert self.tokenizer.sep_token is not None

        return self.tokenizer.tokenize(texts)


class HFListTokenizer(HFTokenizerBase[List[List[str]]]):
    """Process list of texts by separating them by a separator token"""

    def tokenize(self, text_lists: List[List[str]]) -> TokenizedTexts:
        assert self.tokenizer.sep_token is not None
        sep = f" {self.tokenizer.cls_token} "

        return self.tokenizer.tokenize(
            [sep.join(text_list) for text_list in text_lists]
        )
