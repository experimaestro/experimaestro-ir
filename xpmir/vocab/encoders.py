from typing import List, Tuple
import torch.nn as nn
from experimaestro import Config


class Encoder(Config, nn.Module):
    def initialize(self):
        pass

    def static(self):
        return True


class TextEncoder(Encoder):
    """Dense representation of one text"""

    @property
    def dimension(self):
        raise NotImplementedError()


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
