from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, TypeVar, Generic
from attr import define
import re

import torch

from experimaestro import Config
from xpmir.text.utils import lengthToMask
from xpmir.learning.optim import ModuleInitOptions
from xpmir.utils.utils import Initializable
from xpmir.utils.misc import opt_slice
from xpmir.utils.torch import to_device


class TokenizedTexts(NamedTuple):
    """Tokenized texts output"""

    tokens: Optional[List[List[str]]]
    """The list of tokens"""

    ids: torch.LongTensor
    """A matrix containing the token IDs"""

    lens: List[int]
    """the lengths of each text (in tokens)"""

    mask: Optional[torch.LongTensor]
    """The mask for the ids matrix"""

    token_type_ids: Optional[torch.LongTensor] = None
    """Type of each token"""

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ix):
        return TokenizedTexts(
            [opt_slice(self.tokens, i) for i in ix],
            self.ids[ix],
            [self.lens[i] for i in ix],
            opt_slice(self.mask, ix),
            opt_slice(self.token_type_ids, ix),
        )

    def to(self, device: torch.device):
        if device is self.ids.device:
            return self

        return TokenizedTexts(
            self.tokens,
            self.ids.to(device),
            self.lens,
            to_device(self.mask, device),
            to_device(self.token_type_ids, device),
        )


class Tokenizer(Config):
    """
    Represents a vocabulary and a tokenization method

    **Deprecated**: Use TokenizerBase instead
    """

    def tokenize(self, text: str):
        """
        Meant to be overwritten in to provide vocab-specific tokenization when necessary
        e.g., BERT's WordPiece tokenization
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9]", " ", text)
        return text.split()

    def pad_sequences(self, tokensList: List[List[int]], batch_first=True, maxlen=0):
        padding_value = 0
        lens = [len(s) for s in tokensList]
        if maxlen is None:
            maxlen = max(lens)
        else:
            maxlen = min(maxlen or 0, max(lens))

        if batch_first:
            out_tensor = torch.full(
                (len(tokensList), maxlen), padding_value, dtype=torch.long
            )
            for i, tokens in enumerate(tokensList):
                out_tensor[i, : lens[i], ...] = torch.LongTensor(tokens[:maxlen])
        else:
            out_tensor = torch.full(
                (maxlen, len(tokensList)), padding_value, dtype=torch.long
            )
            for i, tokens in enumerate(tokensList):
                out_tensor[: lens[i], i, ...] = tokens[:maxlen]

        return out_tensor.to(self._dummy_params.device), lens

    def batch_tokenize(
        self, texts: List[str], batch_first=True, maxlen=None, mask=False
    ) -> TokenizedTexts:
        """
        Returns tokenized texts

        Arguments:
            mask: Whether a mask should be computed
        """
        toks = [self.tokenize(text) for text in texts]
        tokids, lens = self.pad_sequences(
            [[self.tok2id(t) for t in tok] for tok in toks],
            batch_first=batch_first,
            maxlen=maxlen,
        )

        _mask = lengthToMask(torch.LongTensor(lens)) if mask else None

        return TokenizedTexts(toks, tokids, lens, _mask)

    @property
    def pad_tokenid(self) -> int:
        raise NotImplementedError()

    def tok2id(self, tok: str) -> int:
        """
        Converts a token to an integer id
        """
        raise NotImplementedError()

    def id2tok(self, idx: int) -> str:
        """
        Converts an integer id to a token
        """
        raise NotImplementedError()

    def lexicon_size(self) -> int:
        """
        Returns the number of items in the lexicon
        """
        raise NotImplementedError()


TokenizerInput = TypeVar("TokenizerInput")
TokenizerOutput = TypeVar("TokenizerOutput", bound=TokenizedTexts)


@define
class TokenizerOptions:
    max_length: Optional[int] = None
    return_mask: Optional[bool] = True
    return_length: Optional[bool] = True


class TokenizerBase(
    Config, Initializable, Generic[TokenizerInput, TokenizerOutput], ABC
):
    """Base tokenizer"""

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)

    @abstractmethod
    def tokenize(
        self, inputs: TokenizerInput, options: Optional[TokenizerOptions] = None
    ) -> TokenizerOutput:
        """Encodes the inputs"""
        ...

    @abstractmethod
    def vocabulary_size(self) -> int:
        """
        Returns the number of tokens
        """
        ...

    @abstractmethod
    def tok2id(self, tok: str) -> int:
        """
        Converts a token to an integer id
        """
        ...

    @abstractmethod
    def id2tok(self, idx: int) -> str:
        """
        Converts an integer id to a token
        """
        ...
