from typing import List, Tuple
import torch.nn as nn
import torch
from experimaestro import Config


class Encoder(Config, nn.Module):
    def initialize(self):
        pass

    def static(self):
        return True


class TextEncoder(Encoder):
    """Vector representation of a text - can be dense or sparse"""

    @property
    def dimension(self):
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
        raise NotImplementedError(f"forward in {self.__clas__}")
