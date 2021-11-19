import sys
from typing import Dict, Iterator, Tuple
import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F
from experimaestro import Config, default, Annotated, Param
from xpmir.letor.records import Document, PairwiseRecord, PairwiseRecords, Query
from xpmir.letor.trainers import MetricAccumulator, Trainer
import numpy as np


class PairwiseLoss(Config, nn.Module):
    NAME = "?"

    def compute(self, rel_scores_by_record: Tensor) -> Tuple[Tensor, Dict[str, Tensor]]:
        """
        Compute the loss

        Arguments:

        - rel_scores_by_record: A (batch x 2) tensor (positive/negative)
        """
        raise NotImplementedError()


class CrossEntropyLoss(PairwiseLoss):
    NAME = "cross-entropy"

    def compute(self, rel_scores_by_record):
        target = (
            torch.zeros(rel_scores_by_record.shape[0])
            .long()
            .to(rel_scores_by_record.device)
        )
        return F.cross_entropy(rel_scores_by_record, target, reduction="mean"), {}


class NogueiraCrossEntropyLoss(PairwiseLoss):
    NAME = "nogueira-cross-entropy"

    def compute(self, rel_scores_by_record):
        """
        cross entropy loss formulation for BERT from:
        > Rodrigo Nogueira and Kyunghyun Cho. 2019.Passage re-ranking with bert. ArXiv,
        > abs/1901.04085.
        """
        log_probs = -rel_scores_by_record.log_softmax(dim=2)
        return (log_probs[:, 0, 0] + log_probs[:, 1, 1]).mean(), {}


class SoftmaxLoss(PairwiseLoss):
    """Contrastive loss"""

    NAME = "softmax"

    def compute(self, rel_scores_by_record):
        return torch.mean(1.0 - F.softmax(rel_scores_by_record, dim=1)[:, 0]), {}


class HingeLoss(PairwiseLoss):
    NAME = "hinge"

    margin: Param[float] = 1.0

    def compute(self, rel_scores_by_record):
        return (
            F.relu(
                self.margin - rel_scores_by_record[:, :1] + rel_scores_by_record[:, 1:]
            ).mean(),
            {},
        )


class PointwiseCrossEntropyLoss(PairwiseLoss):
    """Regular PCE (>0 for relevant, 0 otherwise)

    The score of a document is sigmoid(...)"""

    NAME = "pointwise-cross-entropy"

    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()

    def compute(self, rel_scores_by_record):
        device = rel_scores_by_record.device
        dim = rel_scores_by_record.shape[0]
        target = torch.cat(
            (torch.ones(dim, device=device), torch.zeros(dim, device=device))
        )
        return self.loss(rel_scores_by_record.T.flatten(), target), {}


class PairwiseTrainer(Trainer):
    """Pairwse trainer

    Arguments:

    lossfn: The loss function to use
    """

    lossfn: Annotated[PairwiseLoss, default(SoftmaxLoss())]

    def initialize(self, random: np.random.RandomState, ranker, context):
        super().initialize(random, ranker, context)

        self.train_iter_core = self.sampler.pairwiserecord_iter()
        self.train_iter = self.iter_batches(self.train_iter_core)

    def to(self, device):
        self.lossfn.to(device)

    def iter_batches(self, it: Iterator[PairwiseRecord]):
        while True:
            batch = PairwiseRecords()
            for _, record in zip(range(self.batch_size), it):
                batch.add(record)
            yield batch

    def train_batch(self, records: PairwiseRecords, metrics: MetricAccumulator):
        # Get the next batch and compute the scores for each query/document
        rel_scores = self.ranker(records)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        pairwise_scores = rel_scores.reshape(2, len(records)).T
        loss, other_losses = self.lossfn.compute(pairwise_scores)

        losses = {
            f"pairwise-{self.lossfn.NAME}": float(loss.item()),
            "acc": float(self.acc(pairwise_scores).item()),
        }

        for key, value in other_losses.items():
            losses[f"pairwise-{key}"] = float(value.item())

        metrics.update(losses, len(records))

        return loss

    def acc(self, scores_by_record) -> Tensor:
        with torch.no_grad():
            count = scores_by_record.shape[0] * (scores_by_record.shape[1] - 1)
            return (
                scores_by_record[:, :1] > scores_by_record[:, 1:]
            ).sum().float() / count
