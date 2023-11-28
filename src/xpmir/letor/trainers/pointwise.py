from typing import Any
from dataclasses import InitVar
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from experimaestro import Param, Config
from xpmir.letor.records import PointwiseRecords, PointwiseRecord
from xpmir.letor.trainers import LossTrainer
from xpmir.learning.context import Loss, TrainerContext
from xpmir.rankers import LearnableScorer, ScorerOutputType
from xpmir.letor.samplers import PointwiseSampler, SerializableIterator


class PointwiseLoss(Config):
    NAME = "?"
    weight: Param[float] = 1.0

    def initialize(self, scorer: LearnableScorer):
        pass

    def process(self, scores, targets, context: TrainerContext):
        value = self.compute(scores, targets)
        context.add_loss(Loss(f"point-{self.NAME}", value, self.weight))

    def compute(self, rel_scores, target_relscores) -> torch.Tensor:
        raise NotImplementedError()


class MSELoss(PointwiseLoss):
    NAME = "mse"

    def compute(self, rel_scores, target_relscores):
        return F.mse_loss(rel_scores, target_relscores)


class BinaryCrossEntropyLoss(PointwiseLoss):
    """Computes binary cross-entropy

    Uses a BCE with logits if the scorer output type is
    not a probability
    """

    NAME = "bce"

    def initialize(self, scorer: LearnableScorer):
        if scorer.outputType == ScorerOutputType.PROBABILITY:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.BCEWithLogitsLoss()

    def compute(self, rel_scores, target_relscores):
        device = target_relscores.device
        return self.loss(rel_scores.to(device), (target_relscores > 0).float())


class PointwiseTrainer(LossTrainer):
    """Pointwise trainer"""

    lossfn: Param[PointwiseLoss] = MSELoss()
    """Loss function to use"""

    sampler: Param[PointwiseSampler]
    """The pairwise sampler"""

    sampler_iter: InitVar[SerializableIterator[PointwiseRecord, Any]]

    def initialize(self, random: np.random.RandomState, context):
        super().initialize(random, context)

        self.sampler.initialize(self.random)
        self.lossfn.initialize(self.ranker)

        self.random = random
        self.sampler_iter = self.sampler.pointwise_iter()

    def __validate__(self):
        # assert self.grad_acc_batch >= 0, "Adaptative batch size not implemented"
        pass

    def iter_batches(self):
        while True:
            batch = PointwiseRecords()
            for _, record in zip(range(self.batch_size), self.sampler_iter):
                batch.add(record)

            yield batch

    def train_batch(self, records: PointwiseRecords):

        rel_scores = self.ranker(records, self.context)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        target_relscores = torch.FloatTensor(records.relevances)

        rel_scores = rel_scores.flatten()

        self.lossfn.process(rel_scores, target_relscores, self.context)

        # TODO: Create classes for the missing losses
        # Apply the loss
        # elif self.lossfn == "mse-nil":
        #     loss = F.mse_loss(
        #         rel_scores.flatten(), torch.zeros_like(rel_scores.flatten())
        #     )
        # elif self.lossfn == "l1":
        #     loss = F.l1_loss(rel_scores.flatten(), target_relscores)
        # elif self.lossfn == "l1pos":
        #     loss = F.l1_loss(rel_scores.flatten(), (target_relscores > 0.0).float())
        # elif self.lossfn == "smoothl1":
        #     loss = F.smooth_l1_loss(rel_scores.flatten(), target_relscores)
        # elif self.lossfn == "cross_entropy":
        #     loss = -torch.where(
        #         target_relscores > 0, rel_scores.flatten(), 1 - rel_scores.flatten()
        #     ).log()
        #     loss = loss.mean()
        # elif self.lossfn == "cross_entropy_logits":
        #     assert len(rel_scores.shape) == 2
        #     assert rel_scores.shape[1] == 2
        #     log_probs = -rel_scores.log_softmax(dim=1)
        #     one_hot = torch.tensor(
        #         [[1.0, 0.0] if tar > 0 else [0.0, 1.0] for tar in target_relscores],
        #         device=rel_scores.device,
        #     )
        #     loss = (log_probs * one_hot).sum(dim=1).mean()
        # elif self.lossfn == "softmax":
        #     assert len(rel_scores.shape) == 2
        #     assert rel_scores.shape[1] == 2
        #     probs = rel_scores.softmax(dim=1)
        #     one_hot = torch.tensor(
        #         [[0.0, 1.0] if tar > 0 else [1.0, 0.0] for tar in target_relscores],
        #         device=rel_scores.device,
        #     )
        #     loss = (probs * one_hot).sum(dim=1).mean()
        # elif self.lossfn == "mean":
        #     loss = rel_scores.mean()
        # else:
        #     raise ValueError(f"unknown lossfn `{self.lossfn}`")
