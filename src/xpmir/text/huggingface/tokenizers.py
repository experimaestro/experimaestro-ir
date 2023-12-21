import logging
from typing import List, Tuple, Union

import torch
from experimaestro import Param

from xpmir.learning.optim import ModuleInitOptions
from xpmir.text.tokenizers import ListTokenizer, TokenizedTexts, Tokenizer

try:
    from transformers import AutoTokenizer
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise


class HFTokenizer(Tokenizer):
    model_id: Param[str]
    """Model ID from huggingface"""

    def __initialize__(self, options: ModuleInitOptions):
        """Initialize the HuggingFace transformer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        self.cls = self.tokenizer.cls_token
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id

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
        return self.model.config.hidden_size

    @property
    def vocab_size(self) -> int:
        """Returns the size of the vocabulary"""
        return self.tokenizer.vocab_size


class HFListTokenizer(ListTokenizer):
    """Process list of texts by separating them by a separator token"""

    tokenizer: Param[HFTokenizer]
    """The HF tokenizer"""

    def tokenize(self, text_lists: List[List[str]]) -> TokenizedTexts:
        assert self.tokenizer.sep_token is not None
        sep = f" {self.tokenizer.cls_token} "

        self.tokenizer.batch_tokenize([sep.join(text_list) for text_list in text_lists])
