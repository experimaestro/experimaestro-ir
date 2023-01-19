from ast import Dict
import sys
from typing import Iterator, List
import torch
from torch import nn
from torch.functional import Tensor
from experimaestro import Config, Param
from xpmir.letor.records import Document, PairwiseRecord, PairwiseRecords
from xpmir.letor.context import Loss
from xpmir.letor.trainers import TrainerContext, LossTrainer
from xpmir.utils import batchiter, foreach
from .samplers import DistillationPairwiseSampler, PairwiseDistillationSample
import numpy as np
from xpmir.rankers import LearnableScorer


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
        self.lossfn.initialize(self.ranker)
        foreach(
            context.hooks(DistillationPairwiseLoss),
            lambda loss: loss.initialize(self.ranker),
        )
        self.sampler.initialize(random)
        self.train_iter = batchiter(self.batch_size, self.sampler.pairwise_iter())

    # overload the load and write of the state
    # TODO: try to do the same as the original version.
    def load_state_dict(self, state: Dict):
        self.sampler.load_state_dict(state["sampler"])

    def state_dict(self):
        return {"sampler": self.sampler.state_dict()}

    def iter_batches(self) -> Iterator[List[PairwiseDistillationSample]]:
        """Build a iterator over the batches of samples"""
        return self.train_iter

    def train_batch(self, samples: List[PairwiseDistillationSample]):
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
        self.lossfn.process(scores, teacher_scores, self.context)
