from typing import Dict, Iterator
from experimaestro import Param
import numpy as np
from xpmir.learning.context import (
    TrainerContext,
)
from . import Trainer


class MultipleTrainer(Trainer):
    """This trainer can be used to combine various trainers"""

    trainers: Param[Dict[str, Trainer]]
    """The trainers"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        for trainer in self.trainers.values():
            trainer.initialize(random, context)

    def load_state_dict(self, state: Dict):
        for key, trainer in self.trainers.items():
            trainer.load_state_dict(state[key])

    def state_dict(self):
        return {key: trainer.state_dict() for key, trainer in self.trainers.items()}

    def to(self, device):
        """Change the computing device (if this is needed)"""
        super().to(device)
        for trainer in self.trainers.values():
            trainer.to(device)

    def iter_batches(self) -> Iterator:
        iters = {key: trainer.iter_batches() for key, trainer in self.trainers.items()}
        while True:
            yield {key: next(iter) for key, iter in iters.items()}

    def process_batch(self, batch):
        for key, trainer in self.trainers.items():
            with self.context.scope(key):
                trainer.process_batch(batch[key])
