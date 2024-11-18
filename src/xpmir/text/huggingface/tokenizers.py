import os
from pathlib import Path
import logging
from typing import List, Optional, Tuple, Union
from functools import cached_property
import torch
from experimaestro import Config, Param

from xpmir.learning.optim import ModuleInitOptions
from xpmir.utils.utils import Initializable
from xpmir.utils.convert import Converter
from xpmir.text.tokenizers import (
    TokenizerBase,
    TokenizedTexts,
    TokenizerInput,
    TokenizerOptions,
)

try:
    from transformers import AutoTokenizer
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise


HFTokenizerInput = Union[List[str], List[Tuple[str, str]]]


class HFTokenizer(Config, Initializable):
    """This is the main tokenizer class"""

    model_id: Param[str]
    """The tokenizer hugginface ID"""

    max_length: Param[int] = 4096
    """Maximum length for the tokenizer (can be overridden by the model)"""

    DEFAULT_OPTIONS = TokenizerOptions()

    def __initialize__(self, options: ModuleInitOptions):
        """Initialize the HuggingFace transformer"""

        model_id_or_path = self.model_id

        # Use saved models
        if model_path := os.environ.get("XPMIR_TRANSFORMERS_CACHE", None):
            path = Path(model_path) / "tokenizers" / Path(self.model_id)
            if path.is_dir():
                logging.warning("Using saved tokenizer from %s", path)
                model_id_or_path = path
            else:
                logging.warning(
                    "Could not find saved tokenizer in %s, using HF loading", path
                )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path, model_max_length=self.max_length
        )

        self.cls = self.tokenizer.cls_token
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id

    def tokenize(
        self,
        texts: HFTokenizerInput,
        options: Optional[TokenizerOptions] = None,
    ) -> TokenizedTexts:
        options = options or HFTokenizer.DEFAULT_OPTIONS
        max_length = options.max_length
        if max_length is None:
            max_length = self.tokenizer.model_max_length
        else:
            max_length = min(max_length, self.maxtokens())

        r = self.tokenizer(
            list(texts),
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
            return_length=options.return_length,
            return_attention_mask=options.return_mask,
        )

        return TokenizedTexts(
            None,
            r["input_ids"],
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

    @property
    def vocab(self) -> dict:
        return self.tokenizer.vocab


class HFTokenizerBase(TokenizerBase[TokenizerInput, TokenizedTexts]):
    """Base class for all Hugging-Face tokenizers"""

    tokenizer: Param[HFTokenizer]
    """The HuggingFace tokenizer"""

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.tokenizer.initialize(options)

    @classmethod
    def from_pretrained_id(cls, hf_id: str, **kwargs):
        return cls(tokenizer=HFTokenizer(model_id=hf_id), **kwargs)

    def vocabulary_size(self):
        return self.tokenizer.vocab_size

    def tok2id(self, tok: str) -> int:
        return self.tokenizer.tok2id(tok)

    def id2tok(self, idx: int) -> str:
        return self.tokenizer.id2tok(idx)
    
    def get_vocabulary(self):
        return self.tokenizer.vocab


class HFStringTokenizer(HFTokenizerBase[HFTokenizerInput]):
    """Process list of texts"""

    def tokenize(
        self, texts: List[HFTokenizerInput], options: Optional[TokenizerOptions] = None
    ) -> TokenizedTexts:
        return self.tokenizer.tokenize(texts, options=options)


class HFTokenizerAdapter(HFTokenizerBase[TokenizerInput]):
    """Process list of texts"""

    converter: Param[Converter[TokenizerInput, HFTokenizerInput]]

    def tokenize(
        self, inputs: List[TokenizerInput], options: Optional[TokenizerOptions] = None
    ) -> TokenizedTexts:
        return self.tokenizer.tokenize(
            [self.converter(input) for input in inputs], options=options
        )


class HFListTokenizer(HFTokenizerBase[List[List[str]]]):
    """Process list of texts by separating them by a separator token"""

    separate_index: Param[bool] = 0
    """Use a tuple until this index"""

    @cached_property
    def sep_string(self):
        token = self.tokenizer.tokenizer.sep_token
        assert token is not None
        return token

    def join(self, text_list: List[str]):
        if self.separate_index > 0:
            r = text_list[: self.separate_index]
            r.extend([self.sep_string.join(text_list[self.separate_index :])])
            return r

        return self.sep_string.join(text_list)

    def tokenize(
        self, text_lists: List[List[str]], options: Optional[TokenizerOptions] = None
    ) -> TokenizedTexts:
        return self.tokenizer.tokenize(
            [self.join(text_list) for text_list in text_lists],
            options=options,
        )
