from typing import Dict, Iterator, List
from experimaestro import Config, Param
import torch.nn as nn
import numpy as np
from xpmir.utils.utils import EasyLogger
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

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        self.random = random
        # Old style (to be deprecated)
        self.ranker = context.state.model
        # Generic style
        self.model = context.state.model
        self.context = context

        foreach(self.hooks, self.context.add_hook)

    def to(self, device):
        """Change the computing device (if this is needed)"""
        foreach(self.context.hooks(nn.Module), lambda hook: hook.to(device))

    def iter_batches(self) -> Iterator:
        raise NotImplementedError

    def process_batch(self, batch):
        raise NotImplementedError()

    def load_state_dict(self, state: Dict):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()
