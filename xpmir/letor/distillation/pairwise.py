import sys
from typing import Dict, Iterator, List, Tuple
import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F
from experimaestro import Config, default, Annotated, Param, deprecate
from xpmir.letor.records import Document, PairwiseRecord, PairwiseRecords
from xpmir.letor.context import Loss
from xpmir.letor.trainers.pairwise import PairwiseLoss
from xpmir.letor.trainers.pointwise import PointwiseLoss
from xpmir.letor.trainers import TrainerContext, LossTrainer, TrainingHook
from xpmir.utils import batchiter, foreach
from .samplers import DistillationPairwiseSampler, PairwiseDistillationSample
import numpy as np
from xpmir.rankers import LearnableScorer, ScorerOutputType


class DistillationPairwiseLoss(Config, nn.Module):
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

    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def compute(
        self, student_scores: Tensor, teacher_scores: Tensor, info: TrainerContext
    ) -> torch.Tensor:
        return self.loss(
            student_scores[:, 1] - student_scores[:, 0],
            teacher_scores[:, 1] - teacher_scores[:, 0],
        )


class DistillationPairwiseTrainer(LossTrainer):
    """Pairwse trainer

    Arguments:

    lossfn: The loss function to use
    """

    sampler: Param[DistillationPairwiseSampler]

    lossfn: Param[DistillationPairwiseLoss]
    """The distillation pairwise batch function"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.train_iter = batchiter(self.batch_size, self.sampler.pairwise_iter())
        self.lossfn.initialize(self.ranker)

    def train_batch(
        self, samples: List[PairwiseDistillationSample], context: TrainerContext
    ):
        # Builds records and teacher score matrix
        teacher_scores = torch.empty(len(samples), 2)
        records = PairwiseRecords()
        for ix, sample in enumerate(samples):
            records.add(
                PairwiseRecord(
                    sample.query,
                    Document(None, sample.documents[0].content, None),
                    Document(None, sample.documents[1].content, None),
                )
            )
            teacher_scores[ix, 0] = sample.documents[0].score
            teacher_scores[ix, 1] = sample.documents[1].score

        # Get the next batch and compute the scores for each query/document
        scores = self.ranker(records, self.context).reshape(2, len(records)).T

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Call the losses (distillation, pairwise and pointwise)
        teacher_scores = teacher_scores.to(scores.device)
        self.lossfn.process(scores, teacher_scores, context)
