import sys
from abc import ABC, abstractmethod
from dataclasses import InitVar
from typing import Dict

import numpy as np
import torch
from attr import define
from experimaestro import Config, Param

from xpmir.learning.context import Loss
from xpmir.learning.base import BaseSampler
from xpmir.learning import Module
from xpmir.letor.trainers import LossTrainer, TrainerContext
from xpmir.utils.iter import MultiprocessSerializableIterator, SerializableIterator


@define
class RepresentationOutput:
    value: torch.Tensor


class AlignmentLoss(Config, ABC):
    weight: Param[float] = 1.0
    """Weight for this loss"""

    @abstractmethod
    def __call__(
        self,
        input: RepresentationOutput,
        target: RepresentationOutput,
    ):
        """Computes the reconstruction loss

        :param target: a tensor of size BxD
        """
        ...


class MSEAlignmentLoss(AlignmentLoss):
    """Computes the MSE between contextualized query representation and gold
    representation"""

    def __post_init__(self):
        self.mse = torch.nn.MSELoss()

    def __call__(
        self,
        input: RepresentationOutput,
        target: RepresentationOutput,
    ):
        return self.mse(input.value, target.value)


class AlignmentTrainer(LossTrainer):
    """Compares two representations

    Both the representations are expected to a be in a vector space
    """

    losses: Param[Dict[str, AlignmentLoss]]
    """The loss function(s)"""

    sampler: Param[BaseSampler]
    """The pairwise sampler"""

    target_model: Param[Module]
    """Target model"""

    sampler_iter: InitVar[SerializableIterator]
    """The iterator over samples"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.sampler.initialize(random)
        self.sampler_iter = MultiprocessSerializableIterator(
            self.sampler.__batch_iter__(self.batch_size)
        )

    def train_batch(self, batch):
        # Get the next batch and compute the scores for each query/document
        output: RepresentationOutput = self.model(batch)
        target: RepresentationOutput = self.target_model(batch)

        if torch.isnan(output.value).any() or torch.isinf(output.value).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        for key, loss in self.losses.items():
            value = loss(output, target)
            self.context.add_loss(Loss(key, value, loss.weight))
