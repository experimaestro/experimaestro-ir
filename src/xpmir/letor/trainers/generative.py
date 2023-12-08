from torch import nn
import numpy as np
from experimaestro import Param, Config
import logging

from xpmir.letor.samplers import PairwiseSampler
from xpmir.letor.records import BaseRecords, PairwiseRecords
from xpmir.letor.trainers import TrainerContext, LossTrainer
from xpmir.learning.context import Loss
from xpmir.utils.iter import MultiprocessSerializableIterator
from xpmir.utils.utils import foreach, easylog

logger = easylog()


class PairwiseGenerativeLoss(Config, nn.Module):
    """Generic loss for generative models"""

    NAME = "?"

    weight: Param[float] = 1.0
    """The weight :math:`w` with which the loss is multiplied (useful when
    combining with other ones)"""

    def compute(self, records, context):
        pass

    def process(self, records: BaseRecords, context: TrainerContext):
        value = self.compute(records, context)  # tensor shape [bs]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Loss: {value}")
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))


class GenerativeTrainer(LossTrainer):
    loss: Param[PairwiseGenerativeLoss]

    sampler: Param[PairwiseSampler]
    """The pairwise sampler"""

    def initialize(self, random: np.random.RandomState, context: TrainerContext):
        super().initialize(random, context)
        self.loss.initialize()
        foreach(
            context.hooks(PairwiseGenerativeLoss), lambda loss: loss.initialize()
        )  # maybe later we need to change the sampling target, we can use this hook

        self.sampler.initialize(random)
        self.sampler_iter = MultiprocessSerializableIterator(
            self.sampler.pairwise_batch_iter(self.batch_size)
        )

    def train_batch(self, records: PairwiseRecords):
        # do the forward pass to get the gradient value
        self.loss.process(records, self.context)
