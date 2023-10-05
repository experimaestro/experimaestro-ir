import sys
from typing import List
import torch
from torch import nn
from torch.functional import Tensor
from experimaestro import Config, Param
from xpmir.letor.records import (
    DocumentRecord,
    TopicRecord,
    PairwiseRecord,
    PairwiseRecords,
)
from xpmir.learning.context import Loss
from xpmir.letor.trainers import TrainerContext, LossTrainer
from xpmir.utils.utils import foreach
from .samplers import DistillationPairwiseSampler, PairwiseDistillationSample
from xpmir.utils.iter import MultiprocessSerializableIterator
import numpy as np
from xpmir.rankers import LearnableScorer


class DistillationPairwiseLoss(Config, nn.Module):
    """The abstract loss for pairwise distillation"""

    weight: Param[float] = 1.0
    NAME = "?"

    def initialize(self, ranker: LearnableScorer):
        pass

    def process(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ):
        loss = self.compute(student_scores, teacher_scores, info)
        info.add_loss(Loss(f"pairwise-{self.NAME}", loss, self.weight))

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, context: TrainerContext
    ) -> torch.Tensor:
        """
        Compute the loss

        Arguments:

            student_scores: A (batch x 2) tensor
            teacher_scores: A (batch x 2) tensor
        """
        raise NotImplementedError()


class MSEDifferenceLoss(DistillationPairwiseLoss):
    """Computes the MSE between the score differences

    Compute ((student 1 - student 2) - (teacher 1 - teacher 2))**2
    """

    NAME = "delta-MSE"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.loss = nn.MSELoss()

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ) -> torch.Tensor:
        return self.loss(
            student_scores[:, 1] - student_scores[:, 0],
            teacher_scores[:, 1] - teacher_scores[:, 0],
        )


class DistillationKLLoss(DistillationPairwiseLoss):
    """
    Distillation loss from: Distilling Dense Representations for Ranking using
    Tightly-Coupled Teachers https://arxiv.org/abs/2010.11386
    """

    NAME = "Distil-KL"

    def initialize(self, ranker):
        super().initialize(ranker)
        self.loss = nn.KLDivLoss(reduction="none")

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ) -> torch.Tensor:
        pos_student = student_scores[:, 0].unsqueeze(0)
        neg_student = student_scores[:, 1].unsqueeze(0)
        pos_teacher = teacher_scores[:, 0].unsqueeze(0)
        neg_teacher = teacher_scores[:, 1].unsqueeze(0)

        scores = torch.cat([pos_student, neg_student], dim=1)
        local_scores = torch.log_softmax(scores, dim=1)
        teacher_scores = torch.cat(
            [pos_teacher.unsqueeze(-1), neg_teacher.unsqueeze(-1)], dim=1
        )
        teacher_scores = torch.softmax(teacher_scores, dim=1)
        return self.loss(local_scores, teacher_scores).sum(dim=1).mean(dim=0)


class DistillationPairwiseTrainer(LossTrainer):
    """Pairwise trainer for distillation"""

    sampler: Param[DistillationPairwiseSampler]
    """The sampler"""

    lossfn: Param[DistillationPairwiseLoss]
    """The distillation pairwise batch function"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(self.ranker)
        foreach(
            context.hooks(DistillationPairwiseLoss),
            lambda loss: loss.initialize(self.ranker),
        )
        self.sampler.initialize(random)
        self.sampler_iter = self.sampler.pairwise_iter()

        self.sampler_iter = MultiprocessSerializableIterator(
            self.sampler.pairwise_batch_iter(self.batch_size)
        )

    def train_batch(self, samples: List[PairwiseDistillationSample]):
        # Builds records and teacher score matrix
        teacher_scores = torch.empty(len(samples), 2)
        records = PairwiseRecords()
        for ix, sample in enumerate(samples):
            records.add(
                PairwiseRecord(
                    TopicRecord(sample.query),
                    DocumentRecord(sample.documents[0].document),
                    DocumentRecord(sample.documents[1].document),
                )
            )
            teacher_scores[ix, 0] = sample.documents[0].score
            teacher_scores[ix, 1] = sample.documents[1].score

        # Get the next batch and compute the scores for each query/document
        scores = self.ranker(records, self.context).reshape(2, len(records)).T

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            self.logger.error(
                "nan or inf relevance score detected. Aborting (pairwise distillation)."
            )
            sys.exit(1)

        # Call the losses (distillation, pairwise and pointwise)
        teacher_scores = teacher_scores.to(scores.device)
        self.lossfn.process(scores, teacher_scores, self.context)
