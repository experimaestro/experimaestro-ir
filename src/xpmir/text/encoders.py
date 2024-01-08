from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, TypeVar, Union
import sys

from experimaestro import Param
from attrs import define
import torch
import torch.nn as nn

from xpmir.learning.optim import Module
from xpmir.utils.utils import EasyLogger
from .tokenizers import Tokenizer, TokenizedTexts, TokenizerBase, TokenizerOutput


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
    """(deprecated) Represent a text as a sequence of token representations"""

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

    def maxtokens(self) -> int:
        """Maximum number of tokens that can be processed"""
        return sys.maxsize


LegacyEncoderInput = Union[List[str], List[Tuple[str, str]], List[Tuple[str, str, str]]]


InputType = TypeVar("InputType")
EncoderOutput = TypeVar("EncoderOutput")


class TextEncoderBase(Encoder, Generic[InputType, EncoderOutput]):
    """Base class for all text encoders"""

    @property
    def dimension(self) -> int:
        """Returns the dimension of the output space"""
        raise NotImplementedError()

    @abstractmethod
    def forward(self, texts: List[InputType]) -> torch.Tensor:
        raise NotImplementedError()


class TextEncoder(TextEncoderBase[str, torch.Tensor]):
    """Encodes a text into a vector"""

    pass


class DualTextEncoder(TextEncoderBase[Tuple[str, str], torch.Tensor]):
    """Encodes a pair of text into a vector"""

    pass


class TripletTextEncoder(TextEncoderBase[Tuple[str, str, str], torch.Tensor]):
    """Encodes a triplet of text into a vector

    This is used in models such as DuoBERT where we encode (query, positive,
    negative) triplets.
    """

    pass


# --- Generic tokenized text encoders


class TokensRepresentationOutput:
    tokenized: TokenizedTexts
    """Tokenized texts"""

    value: torch.Tensor
    """A 3D tensor (batch x tokens x dimension)"""


class TextsRepresentationOutput:
    tokenized: TokenizedTexts
    """Tokenized texts"""

    value: torch.Tensor
    """A 2D tensor representing full texts (batch x dimension)"""


class TokenizedEncoder(Encoder, Generic[EncoderOutput, TokenizerOutput]):
    """Encodes a tokenized text into a vector"""

    @abstractmethod
    def forward(self, inputs: TokenizerOutput) -> EncoderOutput:
        pass


class TokenizedTextEncoder(
    TextEncoderBase[InputType, EncoderOutput],
    Generic[InputType, EncoderOutput, TokenizerOutput],
):
    """Encodes a tokenizer input into a vector

    This pipelines two objects:

    1. A tokenizer that segments the text;
    2. An encoder that returns a representation of the tokens in a vector space
    """

    tokenizer: Param[TokenizerBase[InputType, TokenizerOutput]]
    encoder: Param[TokenizedEncoder[TokenizerOutput, EncoderOutput]]

    def forward(self, inputs: List[InputType]):
        tokenized = self.tokenizer.tokenize(inputs)
        return self.encoder(tokenized)
