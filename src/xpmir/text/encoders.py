from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, TypeVar, Union
import sys

from attrs import define
from experimaestro import Param
import torch
import torch.nn as nn

from xpmir.learning.optim import Module

from xpmir.utils.utils import EasyLogger
from .tokenizers import (
    Tokenizer,
    TokenizedTexts,
    SimpleTokenizer,
    DualTokenizer,
    TripletTokenizer,
    ListTokenizer,
)

T = TypeVar("T")


class Encoder(Module, EasyLogger, ABC):
    """Base class for all word and text encoders"""

    def __initialize__(self, options):
        # Easy and hacky way to get the device
        super().__initialize__(options)
        self._dummy_params = nn.Parameter(torch.Tensor())

    def static(self):
        return True

    @property
    def device(self):
        return self._dummy_params.device


@define
class TokensEncoderOutput:
    """Output representation for text encoder"""

    tokenized: TokenizedTexts
    """Tokenized texts"""

    value: torch.Tensor
    """The encoder output"""


class TokensEncoder(Tokenizer, Encoder):
    """(deprecated, use TokensEncoderBase) Represent a text as a sequence of
    token representations"""

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

    def forward(self, tokenized: TokenizedTexts):
        """
        Returns embeddings for the tokenized texts.

        tokenized: tokenized texts
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

    def maxtokens(self) -> int:
        """Maximum number of tokens that can be processed"""
        return sys.maxsize


class TokensEncoderBase(Encoder, ABC):
    """Represent a text as a sequence of token representations"""

    def forward(self, texts: TokenizedTexts) -> TokensEncoderOutput:
        ...


LegacyEncoderInput = Union[List[str], List[Tuple[str, str]], List[Tuple[str, str, str]]]


InputType = TypeVar("InputType")


class TextEncoderBase(Encoder, Generic[InputType]):
    """Base class for legacy text encoders"""

    @property
    def dimension(self) -> int:
        """Returns the dimension of the output space"""
        raise NotImplementedError()

    @abstractmethod
    def forward(self, texts: InputType) -> torch.Tensor:
        raise NotImplementedError()


class TextEncoder(TextEncoderBase[str]):
    """Encodes a text into a vector"""

    pass


class DualTextEncoder(TextEncoderBase[Tuple[str, str]]):
    """Encodes a pair of text into a vector"""

    pass


class TripletTextEncoder(TextEncoderBase[Tuple[str, str, str]]):
    """Encodes a triplet of text into a vector

    This is used in models such as DuoBERT where we encode (query, positive,
    negative) triplets.
    """

    pass


class ListTextEncoder(TextEncoderBase[List[str]]):
    """Encodes a list of strings (variable length) into a vector"""

    pass


# --- Tokenized text encoders


class TokenizedTextEncoder(Encoder):
    """Encodes a tokenized text into a vector"""

    @abstractmethod
    def forward(self, inputs: TokenizedTexts) -> torch.Tensor:
        pass


class SimpleTokenizedTextEncoder(TextEncoderBase[str]):
    """Encodes a text into a vector"""

    tokenizer: Param[SimpleTokenizer]
    encoder: Param[TokenizedTextEncoder]

    def forward(self, inputs: List[str]):
        tokenized = self.tokenizer(inputs)
        return self.encoder(tokenized)


class DualTokenizedTextEncoder(TextEncoderBase[Tuple[str, str]]):
    """Encodes a pair of text into a vector"""

    tokenizer: Param[DualTokenizer]
    encoder: Param[TokenizedTextEncoder]

    def forward(self, inputs: List[Tuple[str, str]]):
        tokenized = self.tokenizer(inputs)
        return self.encoder(tokenized)


class TripletTokenizedTextEncoder(TextEncoderBase[Tuple[str, str, str]]):
    """Encodes a triplet of text into a vector

    This is used in models such as DuoBERT where we encode (query, positive,
    negative) triplets.
    """

    tokenizer: Param[TripletTokenizer]
    encoder: Param[TokenizedTextEncoder]

    def forward(self, inputs: List[Tuple[str, str, str]]):
        tokenized = self.tokenizer(inputs)
        return self.encoder(tokenized)


class ListTokenizedTextEncoder(TextEncoderBase[List[str]]):
    """Encodes a list of strings (variable length) into a vector"""

    tokenizer: Param[ListTokenizer]
    encoder: Param[TokenizedTextEncoder]

    def forward(self, inputs: List[List[str]]):
        tokenized = self.tokenizer(inputs)
        return self.encoder(tokenized)
