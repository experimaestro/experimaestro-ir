from dataclasses import InitVar
import sys
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F
from xpmir.learning.context import Loss, TrainerContext
from xpmir.letor.records import (
    MaskedLanguageModelingRecords,
    MaskedLanguageModelingRecord,
)
from xpmir.letor.trainers import LossTrainer

from experimaestro import Param, Config

from xpmir.letor.samplers import Sampler
from xpmir.utils.iter import RandomSerializableIterator


class MLMLoss(Config):
    NAME = "?"
    weight: Param[float] = 1.0

    def initialize(self):
        pass

    def process(self, scores, targets, context: TrainerContext):
        value = self.compute(scores, targets)
        context.add_loss(Loss(f"point-{self.NAME}", value, self.weight))

    def compute(self, rel_scores, target_relscores) -> torch.Tensor:
        raise NotImplementedError()


class CrossEntropyLoss(MLMLoss):
    """Computes cross-entropy

    Uses a CE with logits if the scorer output type is
    not a probability
    """

    NAME = "ce"

    def compute(self, rel_scores, target_relscores):
        return F.cross_entropy(rel_scores, target_relscores)


class MLMTrainer(LossTrainer):
    """Trainer for Masked Language Modeling"""

    # Loss function to use
    lossfn: Param[MLMLoss] = CrossEntropyLoss()

    sampler: Param[Sampler]

    sampler_iter: InitVar[RandomSerializableIterator[MaskedLanguageModelingRecord]]

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize()
        self.sampler.initialize(random)
        self.sampler_iter = self.sampler.record_iter()

    def __validate__(self):
        # assert self.grad_acc_batch >= 0, "Adaptative batch size not implemented"
        pass

    def iter_batches(self) -> Iterator[MaskedLanguageModelingRecords]:
        while True:
            batch = MaskedLanguageModelingRecords()
            for _, record in zip(range(self.batch_size), self.sampler_iter):
                batch.add(record)

            yield batch

    def train_batch(self, records: MaskedLanguageModelingRecords):
        mlm_output = self.model(records.to_texts(), self.context)

        if torch.isnan(mlm_output.logits).any() or torch.isinf(mlm_output.logits).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        self.lossfn.process(
            mlm_output.logits.view(-1, self.model.vocab_size),
            mlm_output.labels.view(-1).to(mlm_output.logits.device),
            self.context,
        )
