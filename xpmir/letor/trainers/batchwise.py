import sys
import torch
import torch.nn.functional as F
from experimaestro import Config, default, Annotated, Param
from xpmir.letor.samplers import BatchwiseSampler
from xpmir.rankers import LearnableScorer
from xpmir.letor.trainers import TrainContext, Trainer
import numpy as np


class BatchwiseLoss(Config):
    NAME = "?"

    def compute(self, scores: torch.Tensor, relevances: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss

        Arguments:

        - rel_scores_by_record: A (batch x 2) tensor (positive/negative)
        """
        raise NotImplementedError()


class CrossEntropyLoss(BatchwiseLoss):
    NAME = "cross-entropy"

    def compute(self, scores, relevances):
        return F.binary_cross_entropy(scores, relevances, reduction="mean")


class ContrastiveLoss:
    """Computes the probability of relevant documents for a given query"""

    def compute(self, score, relevances):
        raise NotImplementedError()


class BatchwiseTrainer(Trainer):
    """Batchwise trainer

    Arguments:

    lossfn: The loss function to use
    sampler: A batchwise sampler
    """

    lossfn: Annotated[BatchwiseLoss, default(CrossEntropyLoss())]
    sampler: Param[BatchwiseSampler]

    def initialize(
        self,
        random: np.random.RandomState,
        ranker: LearnableScorer,
        context: TrainContext,
    ):
        super().initialize(random, ranker, context)

        self.train_iter = iter(self.sampler)

    def train_batch(self):
        # Get the next batch and compute the scores for each query/document
        batch = next(self.train_iter)

        # Get the scores
        rel_scores = self.ranker(batch)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss

        batch_scores = rel_scores.reshape(*batch.relevances.shape)
        loss = self.lossfn.compute(batch_scores, batch.relevances)

        return loss, {f"batchwise-{self.lossfn.NAME}": loss.item()}
