from typing import List, NamedTuple, Optional, Tuple
import torch
import torch.nn as nn
from experimaestro import Param
from xpmir.letor.optim import Module
from . import Vocab


class Encoder(Module):
    """Base class for all word and text encoders"""

    def initialize(self):
        pass

    def static(self):
        return True


class TextEncoder(Encoder):
    """Vector representation of a text - can be dense or sparse"""

    @property
    def dimension(self) -> int:
        """Returns the dimension of the representation"""
        raise NotImplementedError(f"dimension for {self.__class__}")

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a matrix encoding the provided texts"""
        raise NotImplementedError(f"forward for {self.__class__}")


class DualTextEncoder(Encoder):
    """Dense representation for a pair of text

    This is used for instance in the case of BERT models
    that represent the (query, document couple)
    """

    @property
    def dimension(self) -> int:
        raise NotImplementedError()

    def forward(self, texts: List[Tuple[str, str]]):
        """Computes the representation of a list of pair of texts"""
        raise NotImplementedError(f"forward in {self.__class__}")


class TripletTextEncoder(Encoder):
    """The generic class for triplet encoders: query-document-document"""

    @property
    def dimension(self) -> int:
        raise NotImplementedError()

    def forward(self, texts: List[Tuple[str, str, str]]):
        """Computes the representation of a list of pair of texts"""
        raise NotImplementedError(f"forward in {self.__class__}")


class ContextualizedTextEncoderOutput(NamedTuple):
    """The output of a contextualized encoder"""

    ids: torch.Tensor
    """The token IDs"""

    mask: torch.Tensor
    """A 0/1 mask"""

    last_layer: torch.Tensor
    """Last layer"""

    layers: Optional[List[torch.Tensor]]
    """All layers (if asked for)"""


class ContextualizedTextEncoder(Encoder):
    """Returns a contextualized embeddings of tokens"""

    @property
    def dimension(self) -> int:
        raise NotImplementedError()

    @property
    def layers(self) -> int:
        raise NotImplementedError()

    def forward(
        self, texts: List[str], only_tokens=True, output_hidden_states=False
    ) -> ContextualizedTextEncoderOutput:
        """Returns contextualized version of tokens

        Arguments:
            only_tokens: add special tokens to the mask
            all_layers: add all layers to the output
        """
        raise NotImplementedError(f"forward in {self.__clas__}")


class MeanTextEncoder(TextEncoder):
    """Returns the mean of the word embeddings"""

    vocab: Param[Vocab]

    def initialize(self):
        self.vocab.initialize()

    def static(self):
        return self.vocab.static()

    @property
    def dimension(self):
        return self.vocab.dim()

    def forward(self, texts: List[str]) -> torch.Tensor:
        tokenized = self.vocab.batch_tokenize(texts, True)
        emb_texts = self.vocab(tokenized)
        # Computes the mean over the time dimension (vocab output is batch x time x dim)
        return emb_texts.mean(1)
