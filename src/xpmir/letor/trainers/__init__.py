from typing import Dict
from experimaestro import Param
import numpy as np
from xpmir.learning.metrics import ScalarMetric
from xpmir.letor.samplers import Sampler, SerializableIterator
from xpmir.letor.records import BaseRecords
from xpmir.utils.utils import easylog
from xpmir.learning.batchers import Batcher
from xpmir.learning.context import (
    TrainerContext,
)
from xpmir.learning.trainers import Trainer


logger = easylog()


class LossTrainer(Trainer):
    """Trainer based on a loss function

    This trainer supposes that:

    - the `sampler_iter` is initialized – and is a serializable iterator over batches
    """

    batcher: Param[Batcher] = Batcher()
    """How to batch samples together"""

    sampler: Param[Sampler]
    """The sampler to use"""

    batch_size: Param[int] = 16
    """Number of samples per batch"""

    sampler_iter: SerializableIterator
    """The iterator over batches"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)

        self.sampler.initialize(random)

        self.batcher_worker = self.batcher.initialize(self.batch_size)

    def iter_batches(self) -> SerializableIterator:
        """Returns the batchwise iterator"""
        return self.sampler_iter

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
