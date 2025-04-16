from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, TypeVar, Union, Optional, Callable
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
    TokenizerBase,
    TokenizerOutput,
    TokenizerOptions,
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

    def maxtokens(self) -> Optional[int]:
        """Maximum number of tokens that can be processed"""
        return None


LegacyEncoderInput = Union[List[str], List[Tuple[str, str]], List[Tuple[str, str, str]]]


InputType = TypeVar("InputType")
EncoderOutput = TypeVar("EncoderOutput")


class TextEncoderBase(Encoder, Generic[InputType, EncoderOutput]):
    """Base class for all text encoders"""

    __call__: Callable[Tuple["TextEncoderBase", List[InputType]], EncoderOutput]

    @abstractmethod
    def forward(self, texts: InputType) -> EncoderOutput:
        raise NotImplementedError()

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the dimension of the output space"""
        raise NotImplementedError()

    def max_tokens(self):
        """Returns the maximum number of tokens this encoder can process"""
        return sys.maxsize


class TextEncoder(TextEncoderBase[str, torch.Tensor]):
    """Encodes a text into a vector

    .. deprecated:: 1.3
        Use TextEncoderBase directly
    """

    pass


class DualTextEncoder(TextEncoderBase[Tuple[str, str], torch.Tensor]):
    """Encodes a pair of text into a vector

    .. deprecated:: 1.3
        Use TextEncoderBase directly
    """

    pass


class TripletTextEncoder(TextEncoderBase[Tuple[str, str, str], torch.Tensor]):
    """Encodes a triplet of text into a vector

    .. deprecated:: 1.3
        Use TextEncoderBase directly

    This is used in models such as DuoBERT where we encode (query, positive,
    negative) triplets.
    """

    pass


# --- Generic tokenized text encoders


@define
class RepresentationOutput:
    value: torch.Tensor
    """An arbitrary representation (by default, the batch dimension is the
    first)"""

    def __len__(self):
        return len(self.value)

    def __getitem__(self, ix: Union[slice, int]):
        return self.__class__(self.value[ix])

    @property
    def device(self):
        return self.value.device


@define
class TokensRepresentationOutput(RepresentationOutput):
    """A 3D tensor (batch x tokens x dimension)"""

    tokenized: TokenizedTexts
    """Tokenized texts"""

    def __getitem__(self, ix: Union[slice, int]):
        return self.__class__(self.value[ix], self.tokenized[ix])


@define
class TextsRepresentationOutput(RepresentationOutput):
    """Value is atensor representing full texts (batch x dimension)"""

    tokenized: TokenizedTexts
    """Tokenized texts"""

    def to(self, device):
        return self.__class__(self.value.to(device), self.tokenized.to(device))

    def __getitem__(self, ix: Union[slice, int]):
        return self.__class__(self.value[ix], self.tokenized[ix])


class TokenizedEncoder(Encoder, Generic[EncoderOutput, TokenizerOutput]):
    """Encodes a tokenized text into a vector"""

    @abstractmethod
    def forward(self, inputs: TokenizerOutput) -> EncoderOutput:
        pass

    @property
    def max_length(self):
        """Returns the maximum length that the model can process"""
        return sys.maxsize


class TokenizedTextEncoderBase(TextEncoderBase[InputType, EncoderOutput]):
    @abstractmethod
    def forward(
        self, inputs: List[InputType], options: Optional[TokenizerOptions] = None
    ) -> EncoderOutput:
        ...


class TokenizedTextEncoder(
    TokenizedTextEncoderBase[InputType, EncoderOutput],
    Generic[InputType, EncoderOutput, TokenizerOutput],
):
    """Encodes a tokenizer input into a vector

    This pipelines two objects:

    1. A tokenizer that segments the text;
    2. An encoder that returns a representation of the tokens in a vector space
    """

    tokenizer: Param[TokenizerBase[InputType, TokenizerOutput]]
    encoder: Param[TokenizedEncoder[TokenizerOutput, EncoderOutput]]

    def __initialize__(self, options):
        super().__initialize__(options)
        self.tokenizer.initialize(options)
        self.encoder.initialize(options)

    def forward(
        self, inputs: List[InputType], *args, options: Optional[TokenizerOptions] = None
    ) -> EncoderOutput:
        assert len(args) == 0, "Unhandled extra arguments"
        tokenized = self.tokenizer.tokenize(inputs, options)
        return self.forward_tokenized(tokenized, *args)

    def forward_tokenized(self, tokenized):
        return self.encoder(tokenized)

    def tokenize(
        self, inputs: List[InputType], options: Optional[TokenizerOptions] = None
    ):
        options = options or TokenizerOptions()
        options.max_length = min(self.encoder.max_length, options.max_length or None)
        return self.tokenizer.tokenize(inputs, options)

    def static(self):
        """Whether embeddings parameters are learnable"""
        return self.encoder.static()

    @property
    def dimension(self):
        return self.encoder.dimension
