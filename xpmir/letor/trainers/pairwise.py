import sys
from typing import Dict, Iterator, Tuple
import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F
from experimaestro import Config, default, Annotated, Param, deprecate
from xpmir.letor.context import Loss, ScalarMetric, TrainingHook
from xpmir.letor.records import PairwiseRecord, PairwiseRecords
from xpmir.letor.samplers import PairwiseSampler
from xpmir.letor.trainers import TrainContext, Trainer
import numpy as np
from xpmir.rankers import LearnableScorer, ScorerOutputType
from xpmir.utils import foreach


class PairwiseLoss(TrainingHook, nn.Module):
    NAME = "?"
    weight: Param[float] = 1.0

    def initialize(self, ranker: LearnableScorer):
        pass

    def process(self, scores: Tensor, context: TrainContext):
        value = self.compute(scores, context)
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))

    def compute(self, scores: Tensor, info: TrainContext) -> Tensor:
        """
        Compute the loss

        Arguments:

        - scores: A (batch x 2) tensor (positive/negative)
        """
        raise NotImplementedError()


class CrossEntropyLoss(PairwiseLoss):
    NAME = "cross-entropy"

    def compute(self, rel_scores_by_record, info: TrainContext):
        target = (
            torch.zeros(rel_scores_by_record.shape[0])
            .long()
            .to(rel_scores_by_record.device)
        )
        return F.cross_entropy(rel_scores_by_record, target, reduction="mean")


class SoftmaxLoss(PairwiseLoss):
    """Contrastive loss"""

    NAME = "softmax"

    def compute(self, rel_scores_by_record, info: TrainContext):
        return torch.mean(1.0 - F.softmax(rel_scores_by_record, dim=1)[:, 0])


class LogSoftmaxLoss(PairwiseLoss):
    """RankNet loss or log-softmax loss

    Classification loss (relevant vs non-relevant) where the logit
    is equal to the difference between the relevant and the non relevant
    document (or equivalently, softmax then mean log probability of relevant documents)

    Reference: C. Burges et al., “Learning to rank using gradient descent,” 2005.
    """

    NAME = "ranknet"

    def compute(self, scores: torch.Tensor, info: TrainContext):
        return -F.logsigmoid(scores[:, 0] - scores[:, 1]).mean()


RanknetLoss = LogSoftmaxLoss


class HingeLoss(PairwiseLoss):
    NAME = "hinge"

    margin: Param[float] = 1.0

    def compute(self, rel_scores_by_record, info: TrainContext):
        return F.relu(
            self.margin - rel_scores_by_record[:, 0] + rel_scores_by_record[:, 1]
        ).mean()


class BCEWithLogLoss(nn.Module):
    def __call__(self, log_probs, info: TrainContext):
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

    def compute(self, rel_scores_by_record, info: TrainContext):
        if self.rankerOutputType == ScorerOutputType.LOG_PROBABILITY:
            return self.loss(rel_scores_by_record, info)

        device = rel_scores_by_record.device
        dim = rel_scores_by_record.shape[0]
        target = torch.cat(
            (torch.ones(dim, device=device), torch.zeros(dim, device=device))
        )
        return self.loss(rel_scores_by_record.T.flatten(), target)


class PairwiseTrainer(Trainer):
    """Pairwise trainer uses samples of the form (query, positive, negative)"""

    sampler: Param[PairwiseSampler]
    """The pairwise sampler"""

    def initialize(
        self,
        random: np.random.RandomState,
        ranker: LearnableScorer,
        context: TrainContext,
    ):
        super().initialize(random, ranker, context)

        self.train_iter_core = self.sampler.pairwise_iter()
        self.train_iter = self.iter_batches(self.train_iter_core)

        foreach(context.hooks(PairwiseLoss), lambda loss: loss.initialize(ranker))

    def iter_batches(self, it: Iterator[PairwiseRecord]):
        while True:
            batch = PairwiseRecords()
            for _, record in zip(range(self.batch_size), it):
                batch.add(record)
            yield batch

    @deprecate
    def lossfn(self, value):
        """Use hooks instead"""
        assert not self.hooks
        self.hooks = [value]

    def train_batch(self, records: PairwiseRecords):
        # Get the next batch and compute the scores for each query/document
        rel_scores = self.ranker(records, self.context)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        pairwise_scores = rel_scores.reshape(2, len(records)).T
        foreach(
            self.context.hooks(PairwiseLoss),
            lambda loss: loss.process(pairwise_scores, self.context),
        )

        self.context.metrics.add(
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
