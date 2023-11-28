from typing import List, Tuple
import sys
import re

import torch
import torch.nn as nn

from experimaestro import Config, Param
from xpmir.learning.optim import Module
from xpmir.letor.records import TokenizedTexts

from xpmir.utils.utils import EasyLogger


class Encoder(Module, EasyLogger):
    """Base class for all word and text encoders"""

    def __initialize__(self):
        # Easy and hacky way to get the device
        super().__initialize__()
        self._dummy_params = nn.Parameter(torch.Tensor())

    def static(self):
        return True

    @property
    def device(self):
        return self._dummy_params.device


def lengthToMask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.
    """
    assert len(length.shape) == 1, "Length shape should be 1 dimensional."
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


class Tokenizer(Config):
    """
    Represents a vocabulary and a tokenization method
    """

    def tokenize(self, text):
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


class TokensEncoder(Tokenizer, Encoder):
    """Represent a text as a sequence of token representations"""

    def enc_query_doc(
        self, queries: List[str], documents: List[str], d_maxlen=None, q_maxlen=None
    ):
        """
        Returns encoded versions of the query and document from two
        list (same size) of queries and documents

        May be overwritten in subclass to provide contextualized representation, e.g.
        joinly modeling query and document representations in BERT.
        """

        tokenized_queries = self.batch_tokenize(queries, maxlen=q_maxlen)
        tokenized_documents = self.batch_tokenize(documents, maxlen=d_maxlen)
        return (
            tokenized_queries,
            self(tokenized_queries),
            tokenized_documents,
            self(tokenized_documents),
        )

    def forward(self, tok_texts: TokenizedTexts):
        """
        Returns embeddings for the tokenized texts.

        tok_texts: tokenized texts
        """
        raise NotImplementedError()

    def emb_views(self) -> int:
        """
        Returns how many "views" are returned by the embedding layer. Most have
        1, but sometimes it's useful to return multiple, e.g., BERT's multiple
        layers
        """
        return 1

    def dim(self) -> int:
        """
        Returns the number of dimensions of the embedding
        """
        raise NotImplementedError(f"for {self.__class__}")

    def static(self) -> bool:
        """
        Returns True if the representations are static, i.e., not trained.
        Otherwise False. This allows models to know when caching is appropriate.
        """
        return True

    def maxtokens(self) -> float:
        """Maximum number of tokens that can be processed"""
        return sys.maxsize


class TextEncoder(Encoder):
    """Vectorial representation of a text - can be dense or sparse"""

    @property
    def dimension(self) -> int:
        """Returns the dimension of the representation"""
        raise NotImplementedError(f"dimension for {self.__class__}")

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a matrix encoding the provided texts"""
        raise NotImplementedError(f"forward for {self.__class__}")


class DualTextEncoder(Encoder):
    """Vectorial representation for a pair of texts

    This is used for instance in the case of BERT models
    that represent the (query, document couple)
    """

    @property
    def dimension(self) -> int:
        raise NotImplementedError()

    def forward(self, texts: List[Tuple[str, str]]) -> torch.Tensor:
        """Computes the representation of a list of pair of texts"""
        raise NotImplementedError(f"forward in {self.__class__}")


class TripletTextEncoder(Encoder):
    """The generic class for triplet encoders: query-document-document


    This encoding is used in models such as DuoBERT that compute
    whether a pair is preferred to another
    """

    @property
    def dimension(self) -> int:
        raise NotImplementedError()

    def forward(self, texts: List[Tuple[str, str, str]]) -> torch.Tensor:
        """Computes the representation of a list of pair of texts"""
        raise NotImplementedError(f"forward in {self.__class__}")


class MeanTextEncoder(TextEncoder):
    """Returns the mean of the word embeddings"""

    encoder: Param[TokensEncoder]

    def __initialize__(self):
        self.encoder.__initialize__()

    def static(self):
        return self.encoder.static()

    @property
    def dimension(self):
        return self.encoder.dim()

    def forward(self, texts: List[str]) -> torch.Tensor:
        tokenized = self.encoder.batch_tokenize(texts, True)
        emb_texts = self.encoder(tokenized)
        # Computes the mean over the time dimension (vocab output is batch x time x dim)
        return emb_texts.mean(1)
