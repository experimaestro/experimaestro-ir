from dataclasses import InitVar
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

    def compute(self, scores: torch.Tensor, info: TrainerContext):
        return -F.logsigmoid(scores[:, 0] - scores[:, 1]).mean()


RanknetLoss = LogSoftmaxLoss


class HingeLoss(PairwiseLoss):
    NAME = "hinge"

    margin: Param[float] = 1.0

    def compute(self, rel_scores_by_record, info: TrainerContext):
        return F.relu(
            self.margin - rel_scores_by_record[:, 0] + rel_scores_by_record[:, 1]
        ).mean()


class BCEWithLogLoss(nn.Module):
    def __call__(self, log_probs, info: TrainerContext):
        assert info.metrics is not None, "No metrics object in context"

        # Assumes target is a two column matrix (rel. / not rel.)
        rel_cost, nrel_cost = (
            -log_probs[:, 0].mean(),
            -(1.0 - log_probs[:, 1].exp()).log().mean(),
        )
        info.metrics.add(
            ScalarMetric("pairwise-pce-rel", rel_cost.item(), len(log_probs))
        )
        info.metrics.add(
            ScalarMetric("pairwise-pce-nrel", nrel_cost.item(), len(log_probs))
        )
        return (rel_cost + nrel_cost) / 2


class PointwiseCrossEntropyLoss(PairwiseLoss):
    """Regular PCE (>0 for relevant, 0 otherwise)
    Uses the ranker output type:
    - If real, uses a BCELossWithLogits (sigmoid transformation)
    - If probability, uses the BCELoss
    - If log probability, uses a custom BCE loss
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
    pass


class PairwiseLossWithTarget(PairwiseLossWithTarget):
    NAME = "DuoLogProbaLoss"

    def compute(self, score: Tensor, target: Tensor, context: TrainerContext):
        return self.loss(score, target)

    def process(self, scores: Tensor, target: Tensor, context: TrainerContext):
        # TODO: adapt the loss to the scorer output (see PairwiseLoss)
        self.loss = nn.BCEWithLogitsLoss()
        value = self.compute(scores, target, context)
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))


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
        foreach(context.hooks(PairwiseLoss), lambda loss: loss.initialize(self.ranker))
        self.sampler.initialize(random)
        self.sampler_iter = self.sampler.pairwise_iter()

    def iter_batches(self) -> Iterator[PairwiseRecordsWithTarget]:
        while True:
            batch = PairwiseRecordsWithTarget()
            for _, record in zip(range(self.batch_size), self.sampler_iter):
                # randomly swap the first and second document Test: some errors
                # maybe related here. modify the 0.5 to 1 to see the source of
                # the error
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
        targets = records.get_target().to(rel_scores.device())

        # print(rel_scores)
        # print(torch.Tensor(records.get_target()))

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
            count = scores_by_record.shape[0]  # batch_size
            pos = scores_by_record > 0
            return (
                torch.logical_not(torch.logical_xor(pos, target))
            ).sum().float() / count
