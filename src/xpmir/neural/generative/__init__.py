from dataclasses import dataclass
from typing import List, Optional
from abc import abstractmethod

import torch
from xpmir.learning.optim import Module
from xpmir.utils.utils import easylog

logger = easylog()


class StepwiseGenerator:
    """Utility class for generating one token at a time"""

    @abstractmethod
    def init(self, texts: List[str]) -> torch.Tensor:
        """Returns the distribution over the first generated tokens (BxV)
        given the texts"""
        pass

    @abstractmethod
    def step(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Returns the distribution over next tokens (BxV), given the last
        generates ones (B)"""
        pass

    @abstractmethod
    def state(self):
        """Get the current state, so we can start back to a previous generated prefix"""
        ...

    @abstractmethod
    def load_state(self, state):
        """Load a saved state"""
        ...


class SequenceGenerator:
    """Utility class for generating a full sequence at a time given the decoder
    input"""

    @abstractmethod
    def init(self, texts: List[str]) -> torch.Tensor:
        """Returns the distribution over the first generated tokens (BxV)
        given the texts"""
        pass

    @abstractmethod
    def decode(
        self,
        labels: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
    ):
        """Returns the logits given the tokens ids and return the logits of
        shape [bs, seq, dec_dim]"""
        pass


@dataclass
class GenerateOptions:
    """Options used during sequence generation"""

    return_dict_in_generate: bool = True
    max_new_tokens: int = 10
    """The number of new tokens to be generated"""
    output_scores: bool = True
    num_return_sequences: int = 1
    """number of returned sequences"""


@dataclass
class BeamSearchGenerationOptions(GenerateOptions):
    """Options related to the beam search of the generate method"""

    num_beams: int = 1
    """beam size"""


class ConditionalGenerator(Module):
    """Models that generate an identifier given a document or a query"""

    @abstractmethod
    def stepwise_iterator(self) -> StepwiseGenerator:
        pass

    @abstractmethod
    def sequence_generator(self) -> SequenceGenerator:
        pass

    @abstractmethod
    def generate(self, inputs: List[str], options: GenerateOptions = None):
        """Generate text given the inputs"""
        pass
