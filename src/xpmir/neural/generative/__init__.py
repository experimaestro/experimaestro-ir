from experimaestro import Param, Config
from typing import List, NamedTuple, Optional
from abc import abstractmethod

import torch
from xpmir.learning.optim import Module
from xpmir.rankers import AbstractModuleScorer
from xpmir.utils.utils import easylog

logger = easylog()


class GeneratorForwardOutput(NamedTuple):
    """The forward output of the generative retrieval"""

    logits: torch.tensor
    past_key_values: Optional[torch.tensor] = None


class DepthUpdatable(Config):
    """Abstract class of the objects which could update their depth"""

    @abstractmethod
    def update_depth(self, new_depth):
        pass


class StepwiseGenerator:
    """Utility class for generating one token at a time"""

    decoder_input_ids: torch.LongTensor
    """The current token"""

    @abstractmethod
    def init(self, texts: List[str]) -> torch.Tensor:
        """Returns the distribution over the first generated tokens (BxV)"""
        pass

    @abstractmethod
    def step(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Returns the distribution over next tokens (BxV), given the last
        generates ones (B)"""
        pass


class IdentifierGenerator(Module):
    """Models that generate an identifier given a document or a query"""

    def __initialize__(self):
        pass

    @abstractmethod
    def stepwise_iterator(self) -> StepwiseGenerator:
        pass


class GenerativeRetrievalScorer(AbstractModuleScorer, DepthUpdatable):
    """The abstract class for the generative retrieval scorer"""

    id_generator: Param[IdentifierGenerator]
    """The id generator"""

    max_depth: Param[int] = 5
    """The max depth we need to consider, counting from 1"""

    current_max_depth: int
    """The max_depth for the current learning stage in the progressive training
    stage"""

    def _initialize(self, random):
        self.id_generator.initialize()
        # if no update or in the final evaluation
        self.current_max_depth = self.max_depth

    def update_depth(self, new_depth):
        if new_depth <= self.max_depth:
            self.current_max_depth = new_depth
            logger.info(
                f"Update the max_depth to {self.current_max_depth} for the scorer"
            )
        else:
            self.current_max_depth = self.max_depth
