from typing import Dict, Iterator, List
from experimaestro import Config, Param
import torch.nn as nn
import numpy as np
from xpmir.letor.metrics import ScalarMetric
from xpmir.letor.samplers import Sampler, SerializableIterator
from xpmir.letor.records import BaseRecords
from xpmir.utils.utils import EasyLogger, easylog
from xpmir.letor.batchers import Batcher
from xpmir.letor.context import (
    TrainingHook,
    TrainerContext,
)

from xpmir.utils.utils import foreach

logger = easylog()


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
        self.ranker = context.state.model
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


class LossTrainer(Trainer):
    """Trainer based on a loss function"""

    batcher: Param[Batcher] = Batcher()
    """How to batch samples together"""

    sampler: Param[Sampler]
    """The sampler to use"""

    batch_size: Param[int] = 16
    """Number of samples per batch (the notion of sample depends on the sampler)"""

    sampler_iter: SerializableIterator
    """The iterator over samples"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)

        self.sampler.initialize(random)

        self.batcher_worker = self.batcher.initialize(self.batch_size)

    def load_state_dict(self, state: Dict):
        self.sampler_iter.load_state_dict(state["sampler"])

    def state_dict(self):
        return {"sampler": self.sampler_iter.state_dict()}

    def process_batch(self, batch: BaseRecords):
        """Called by the learner to process a batch of records"""
        self.batcher_worker.process(batch, self.process_microbatch, raise_oom=True)

    def process_microbatch(self, records: BaseRecords):
        """Combines a forward and backard

        This method can be implemented by specific trainers that use the gradient.
        In that case the regularizer losses should be taken into account with
        `self.add_losses`.
        """
        # Restrict losses to this context
        with self.context.losses() as losses:
            self.train_batch(records)
            nrecords = len(records)
            total_loss = 0.0
            names = []

            for loss in losses:
                total_loss += loss.weight * loss.value
                names.append(loss.name)
                self.context.add_metric(
                    ScalarMetric(f"{loss.name}", float(loss.value.item()), nrecords)
                )

            # Reports the main metric
            if len(names) > 1:
                names.sort()
                self.context.add_metric(
                    ScalarMetric("+".join(names), float(total_loss.item()), nrecords)
                )

            self.context.state.optimizer.scale(total_loss).backward()

    def train_batch(self, records):
        """This method should report"""
        raise NotImplementedError
