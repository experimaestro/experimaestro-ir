from dataclasses import InitVar
import math
import sys
from typing import Iterator
import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F
from experimaestro import Config, Param
from xpmir.letor.context import Loss
from xpmir.letor.metrics import ScalarMetric
from xpmir.letor.records import (
    PairwiseRecord,
    PairwiseRecordWithTarget,
    PairwiseRecords,
    PairwiseRecordsWithTarget,
)
from xpmir.letor.samplers import PairwiseSampler, SerializableIterator
from xpmir.letor.trainers import TrainerContext, LossTrainer
import numpy as np
from xpmir.rankers import LearnableScorer, ScorerOutputType
from xpmir.utils.utils import foreach


class PairwiseLoss(Config, nn.Module):
    NAME = "?"
    weight: Param[float] = 1.0

    def initialize(self, ranker: LearnableScorer):
        pass

    def process(self, scores: Tensor, context: TrainerContext):
        value = self.compute(scores, context)
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))

    def compute(self, scores: Tensor, info: TrainerContext) -> Tensor:
        """
        Compute the loss
        Arguments:
        - scores: A (batch x 2) tensor (positive/negative)
        """
        raise NotImplementedError()


class CrossEntropyLoss(PairwiseLoss):
    NAME = "cross-entropy"

    def compute(self, rel_scores_by_record, info: TrainerContext):
        target = (
            torch.zeros(rel_scores_by_record.shape[0])
            .long()
            .to(rel_scores_by_record.device)
        )
        return F.cross_entropy(rel_scores_by_record, target, reduction="mean")


class SoftmaxLoss(PairwiseLoss):
    """Contrastive loss"""

    NAME = "softmax"

    def compute(self, rel_scores_by_record, info: TrainerContext):
        return torch.mean(1.0 - F.softmax(rel_scores_by_record, dim=1)[:, 0])


class LogSoftmaxLoss(PairwiseLoss):
    """RankNet loss or log-softmax loss
    Classification loss (relevant vs non-relevant) where the logit
    is equal to the difference between the relevant and the non relevant
    document (or equivalently, softmax then mean log probability of relevant documents)
    Reference: C. Burges et al., “Learning to rank using gradient descent,” 2005.
    """

    NAME = "ranknet"

    def __initialize__(self, ranker: LearnableScorer):
        assert (
            ranker.outputType != ScorerOutputType.PROBABILITY
        ), "Probability outputs are not handled"

    def compute(self, scores: torch.Tensor, info: TrainerContext):
        return -F.logsigmoid(scores[:, 0] - scores[:, 1]).mean()


RanknetLoss = LogSoftmaxLoss


class HingeLoss(PairwiseLoss):
    """Hinge loss"""

    NAME = "hinge"

    margin: Param[float] = 1.0
    """The margin for the Hinge loss"""

    def compute(self, rel_scores_by_record, info: TrainerContext):
        return F.relu(
            self.margin - rel_scores_by_record[:, 0] + rel_scores_by_record[:, 1]
        ).mean()


class BCEWithLogLoss(nn.Module):
    """Custom cross-entropy loss when outputs are log probabilities"""

    def __call__(self, log_probs: torch.Tensor, targets: torch.Tensor):
        # Assumes target is a two column matrix (rel. / not rel.)

        loss = (
            -log_probs[targets > 0].sum() + (1.0 - log_probs[targets == 0].exp()).sum()
        )

        return loss / log_probs.numel()


class PointwiseCrossEntropyLoss(PairwiseLoss):
    """Point-wise cross-entropy loss

    This loss adapts to the ranker output type:
    - If real, uses a BCELossWithLogits (sigmoid transformation)
    - If probability, uses the BCELoss
    - If log probability, uses a BCEWithLogLoss
    """

    NAME = "pointwise-cross-entropy"

    def initialize(self, ranker: LearnableScorer):
        super().initialize(ranker)
        self.rankerOutputType = ranker.outputType
        if ranker.outputType == ScorerOutputType.REAL:
            self.loss = nn.BCEWithLogitsLoss()
        elif ranker.outputType == ScorerOutputType.PROBABILITY:
            self.loss = nn.BCELoss()
        elif ranker.outputType == ScorerOutputType.LOG_PROBABILITY:
            self.loss = BCEWithLogLoss()
        else:
            raise Exception("Not implemented")

    def compute(self, rel_scores_by_record, info: TrainerContext):
        if self.rankerOutputType == ScorerOutputType.LOG_PROBABILITY:
            return self.loss(rel_scores_by_record, info)

        device = rel_scores_by_record.device
        dim = rel_scores_by_record.shape[0]
        target = torch.cat(
            (torch.ones(dim, device=device), torch.zeros(dim, device=device))
        )
        return self.loss(rel_scores_by_record.T.flatten(), target)


class PairwiseTrainer(LossTrainer):
    """Pairwise trainer uses samples of the form (query, positive, negative)"""

    lossfn: Param[PairwiseLoss]
    """The loss function"""

    sampler: Param[PairwiseSampler]
    """The pairwise sampler"""

    sampler_iter: InitVar[SerializableIterator[PairwiseRecord]]

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(self.ranker)
        foreach(context.hooks(PairwiseLoss), lambda loss: loss.initialize(self.ranker))
        self.sampler.initialize(random)
        self.sampler_iter = self.sampler.pairwise_iter()

    def iter_batches(self) -> Iterator[PairwiseRecords]:
        while True:
            batch = PairwiseRecords()
            for _, record in zip(range(self.batch_size), self.sampler_iter):
                batch.add(record)
            yield batch

    def train_batch(self, records: PairwiseRecords):
        # Get the next batch and compute the scores for each query/document
        rel_scores = self.ranker(records, self.context)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        pairwise_scores = rel_scores.reshape(2, len(records)).T
        self.lossfn.process(pairwise_scores, self.context)

        self.context.add_metric(
            ScalarMetric(
                "accuracy", float(self.acc(pairwise_scores).item()), len(rel_scores)
            )
        )

    def acc(self, scores_by_record) -> Tensor:
        with torch.no_grad():
            count = scores_by_record.shape[0] * (scores_by_record.shape[1] - 1)
            return (
                scores_by_record[:, :1] > scores_by_record[:, 1:]
            ).sum().float() / count


class PairwiseLossWithTarget(Config):
    NAME = "?"
    weight: Param[float] = 1.0

    def initialize(self, ranker: LearnableScorer):
        pass

    def process(self, scores: Tensor, targets: Tensor, context: TrainerContext):
        value = self.compute(scores, targets, context)
        context.add_loss(Loss(f"duo-{self.NAME}", value, self.weight))


class PairwiseLossWithTarget(PairwiseLossWithTarget):
    NAME = "logproba"

    def initialize(self, ranker: LearnableScorer):
        self.loss = {
            ScorerOutputType.REAL: nn.BCEWithLogitsLoss,
            ScorerOutputType.LOG_PROBABILITY: None,
            ScorerOutputType.PROBABILITY: nn.BCELoss,
        }[ranker.outputType]()

    def compute(self, scores: Tensor, targets: Tensor, context: TrainerContext):
        return self.loss(scores, targets)


class DuoPairwiseTrainer(LossTrainer):
    """The pairwise trainer for duobert. The iter_batch method
    can be the same as the pairwiseTrainer
    """

    lossfn: Param[PairwiseLossWithTarget]
    """The loss function"""

    sampler: Param[PairwiseSampler]
    """The pairwise sampler"""

    sampler_iter: InitVar[SerializableIterator[PairwiseRecord]]

    def initialize(self, random: np.random.RandomState, context: TrainerContext):
        super().initialize(random, context)
        self.lossfn.initialize(self.ranker)

        self.score_threshold = {
            ScorerOutputType.LOG_PROBABILITY: math.log(0.5),
            ScorerOutputType.PROBABILITY: 0.5,
            ScorerOutputType.REAL: 0,
        }[self.ranker.outputType]
        foreach(context.hooks(PairwiseLoss), lambda loss: loss.initialize(self.ranker))
        self.sampler.initialize(random)
        self.sampler_iter = self.sampler.pairwise_iter()

    def iter_batches(self) -> Iterator[PairwiseRecordsWithTarget]:
        while True:
            batch = PairwiseRecordsWithTarget()
            for _, record in zip(range(self.batch_size), self.sampler_iter):
                # randomly swap the first and second document
                if self.random.random() < 0.5:
                    batch.add(
                        PairwiseRecordWithTarget(
                            record.query, record.positive, record.negative, 1
                        )
                    )
                else:
                    batch.add(
                        PairwiseRecordWithTarget(
                            record.query, record.negative, record.positive, 0
                        )
                    )
            yield batch

    def train_batch(self, records: PairwiseRecords):
        # Get the next batch and compute the scores for each query/document
        # forward pass
        rel_scores = self.ranker(records, self.context)  # shape: (bs)
        targets = torch.Tensor(records.get_target()).to(rel_scores.device)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        self.lossfn.process(
            rel_scores,
            torch.Tensor(targets),
            self.context,
        )

        self.context.add_metric(
            ScalarMetric(
                "accuracy",
                self.acc(
                    rel_scores,
                    torch.Tensor(targets),
                ).item(),
                len(rel_scores),
            )
        )

    def acc(self, scores_by_record, target) -> Tensor:
        with torch.no_grad():
            positives = scores_by_record > self.score_threshold
            return (positives == target).sum() / len(positives)
