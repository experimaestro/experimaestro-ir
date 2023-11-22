from abc import abstractmethod
from typing import Dict, Iterator, List, Optional
from experimaestro import Config, Param
import torch.nn as nn
import numpy as np
from xpmir.utils.utils import EasyLogger
from xpmir.learning import Module
from xpmir.learning.context import (
    TrainingHook,
    TrainerContext,
)

from xpmir.utils.utils import foreach


class Trainer(Config, EasyLogger):
    """Generic trainer"""

    hooks: Param[List[TrainingHook]] = []
    """Hooks for this trainer: this includes the losses, but can be adapted for
        other uses

        The specific list of hooks depends on the specific trainer
    """

    model: Param[Optional[Module]] = None
    """If the model to optimize is different from the model passsed to Learn,
    this parameter can be used – initialization is still expected to be done at
    the learner level"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        self.random = random

        # Generic style
        if self.model is None:
            self.model = context.state.model

        # Old style (to be deprecated)
        self.ranker = self.model

        self.context = context

        foreach(self.hooks, self.context.add_hook)

    def to(self, device):
        """Change the computing device (if this is needed)"""
        foreach(self.context.hooks(nn.Module), lambda hook: hook.to(device))

    @abstractmethod
    def iter_batches(self) -> Iterator:
        """Returns a (serializable) iterator over batches"""
        ...

    @abstractmethod
    def process_batch(self, batch):
        ...

    @abstractmethod
    def load_state_dict(self, state: Dict):
        ...

    @abstractmethod
    def state_dict(self):
        ...
