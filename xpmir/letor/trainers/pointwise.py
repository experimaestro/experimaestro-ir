import sys
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from experimaestro import param, config
from xpmir.letor.samplers import Records
from xpmir.letor.trainers import Trainer


@param("lossfn", default="mse")
@config()
class PointwiseTrainer(Trainer):
    def initialize(self, random: np.random.RandomState, ranker):
        super().initialize(random, ranker)

        self.sampler.initialize(self.random)

        self.random = random
        self.train_iter_core = self.sampler.record_iter()
        self.train_iter = self.iter_batches(self.train_iter_core)

    def iter_batches(self, it):
        while True:  # breaks on StopIteration
            batch = Records()
            for _, record in zip(range(self.batch_size), it):
                batch.add(record)

            yield batch

    def train_batch(self):
        # Get the next batch
        batch = next(self.train_iter)

        rel_scores = self.ranker(batch)
        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        target_relscores = batch.relevances.float()
        target_relscores[
            target_relscores == -999.0
        ] = 0.0  # replace -999 with non-relevant score

        # Apply the loss
        if self.lossfn == "mse":
            loss = F.mse_loss(rel_scores.flatten(), target_relscores)
        elif self.lossfn == "mse-nil":
            loss = F.mse_loss(
                rel_scores.flatten(), torch.zeros_like(rel_scores.flatten())
            )
        elif self.lossfn == "l1":
            loss = F.l1_loss(rel_scores.flatten(), target_relscores)
        elif self.lossfn == "l1pos":
            loss = F.l1_loss(rel_scores.flatten(), (target_relscores > 0.0).float())
        elif self.lossfn == "smoothl1":
            loss = F.smooth_l1_loss(rel_scores.flatten(), target_relscores)
        elif self.lossfn == "cross_entropy":
            loss = -torch.where(
                target_relscores > 0, rel_scores.flatten(), 1 - rel_scores.flatten()
            ).log()
            loss = loss.mean()
        elif self.lossfn == "cross_entropy_logits":
            assert len(rel_scores.shape) == 2
            assert rel_scores.shape[1] == 2
            log_probs = -rel_scores.log_softmax(dim=1)
            one_hot = torch.tensor(
                [[1.0, 0.0] if tar > 0 else [0.0, 1.0] for tar in target_relscores],
                device=rel_scores.device,
            )
            loss = (log_probs * one_hot).sum(dim=1).mean()
        elif self.lossfn == "softmax":
            assert len(rel_scores.shape) == 2
            assert rel_scores.shape[1] == 2
            probs = rel_scores.softmax(dim=1)
            one_hot = torch.tensor(
                [[0.0, 1.0] if tar > 0 else [1.0, 0.0] for tar in target_relscores],
                device=rel_scores.device,
            )
            loss = (probs * one_hot).sum(dim=1).mean()
        elif self.lossfn == "mean":
            loss = rel_scores.mean()
        else:
            raise ValueError(f"unknown lossfn `{self.lossfn}`")

        losses = {"data": loss}
        loss_weights = {"data": 1.0}
        return {
            "losses": losses,
            "loss_weights": loss_weights,
        }

    def fast_forward(self, record_count):
        self._fast_forward(self.train_iter_core, self.iter_fields, record_count)
