import sys
from typing import Iterator
import torch
import torch.nn.functional as F
from experimaestro import Config, Param, initializer
from xpmir.letor.samplers import BatchwiseSampler, BatchwiseRecords
from xpmir.learning.context import Loss, TrainerContext
from xpmir.rankers import ScorerOutputType
from xpmir.letor.trainers import LossTrainer
import numpy as np


class BatchwiseLoss(Config):
    NAME = "?"

    weight: Param[float] = 1.0
    """The weight of this loss"""

    def initialize(self, context: TrainerContext):
        pass

    def process(
        self, scores: torch.Tensor, relevances: torch.Tensor, context: TrainerContext
    ):
        value = self.compute(scores, relevances, context)
        context.add_loss(Loss(f"batch-{self.NAME}", value, self.weight))

    def compute(
        self, scores: torch.Tensor, relevances: torch.Tensor, context: TrainerContext
    ) -> torch.Tensor:
        """
        Compute the loss

        Arguments:

        - scores: A (queries x documents) tensor
        - relevances: A (queries x documents) tensor
        """
        raise NotImplementedError()


class CrossEntropyLoss(BatchwiseLoss):
    NAME = "bce"

    def compute(self, scores, relevances, context):
        return F.binary_cross_entropy(scores, relevances, reduction="mean")


class SoftmaxCrossEntropy(BatchwiseLoss):
    NAME = "ce"

    """Computes the probability of relevant documents for a given query"""

    def initialize(self, context: TrainerContext):
        super().initialize(context)
        self.mode = context.state.model.outputType
        self.normalize = {
            ScorerOutputType.REAL: lambda x: F.log_softmax(x, -1),
            ScorerOutputType.LOG_PROBABILITY: lambda x: x,
            ScorerOutputType.PROBABILITY: lambda x: x.log(),
        }[context.state.model.outputType]

    def compute(self, scores, relevances, context):
        return -torch.logsumexp(
            self.normalize(scores) + (1 - 1.0 / relevances), -1
        ).sum() / len(scores)


class BatchwiseTrainer(LossTrainer):
    """Batchwise trainer

    Arguments:

    lossfn: The loss function to use
    sampler: A batchwise sampler
    """

    sampler: Param[BatchwiseSampler]
    """A batch-wise sampler"""

    lossfn: Param[BatchwiseLoss]
    """A batchwise loss function"""

    @initializer
    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(context)
        self.sampler_iter = self.sampler.batchwise_iter(self.batch_size)

    def iter_batches(self) -> Iterator[BatchwiseRecords]:
        return self.sampler_iter

    def train_batch(self, batch: BatchwiseRecords):
        # Get the next batch and compute the scores for each query/document
        # Get the scores
        rel_scores = self.ranker(batch, self.context)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        batch_scores = rel_scores.reshape(*batch.relevances.shape)
        self.lossfn.process(
            batch_scores, batch.relevances.to(batch_scores.device), self.context
        )
