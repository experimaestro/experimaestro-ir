import sys
from abc import ABC, abstractmethod
from dataclasses import InitVar
from typing import Dict, TypeVar, Generic

import numpy as np
import torch
from experimaestro import Config, Param

from xpmir.learning.context import Loss
from xpmir.learning.base import BaseSampler
from xpmir.learning import Module, ModuleInitMode
from xpmir.letor.trainers import LossTrainer, TrainerContext
from xpmir.utils.iter import MultiprocessSerializableIterator, SerializableIterator
from xpmir.text.encoders import RepresentationOutput

AlignementLossInput = TypeVar("AlignementLossInput")
AlignmentLossTarget = TypeVar("AlignmentLossTarget")


class AlignmentLoss(Config, ABC, Generic[AlignementLossInput, AlignmentLossTarget]):
    weight: Param[float] = 1.0
    """Weight for this loss"""

    @abstractmethod
    def __call__(
        self,
        input: AlignementLossInput,
        target: AlignmentLossTarget,
    ):
        """Computes the reconstruction loss

        :param target: a tensor of size BxD
        """
        ...


class MSEAlignmentLoss(AlignmentLoss[RepresentationOutput, RepresentationOutput]):
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


class CosineAlignmentLoss(AlignmentLoss[RepresentationOutput, RepresentationOutput]):
    """Computes the MSE between contextualized query representation and gold
    representation"""

    def __post_init__(self):
        self.loss = torch.nn.CosineEmbeddingLoss()

    def __call__(
        self,
        input: RepresentationOutput,
        target: RepresentationOutput,
    ):
        return self.loss(
            input.value, target.value, torch.Tensor([1]).to(input.value.device)
        )


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
        self.target_model.initialize(ModuleInitMode.DEFAULT.to_options())
        self.target_model.to(context.device_information.device)

    def train_batch(self, batch):
        # Get the next batch and compute the scores for each query/document
        output: RepresentationOutput = self.model(batch)
        with torch.no_grad():
            target: RepresentationOutput = self.target_model(batch)

        if torch.isnan(output.value).any() or torch.isinf(output.value).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        for key, loss in self.losses.items():
            value = loss(output, target)
            self.context.add_loss(Loss(key, value, loss.weight))
