import sys
from typing import List
import torch
from torch import nn
from torch.functional import Tensor
from experimaestro import Config, Param, field
from xpmir.letor.records import (
    DocumentRecord,
    ListwiseRecord,
    ListwiseRecords,
)
from xpm_torch.trainers import TrainerContext, LossTrainer
from xpm_torch.losses import Loss 

from xpmir.utils.utils import foreach
from .samplers import DistillationListwiseSampler
from xpmir.utils.iter import MultiprocessSerializableIterator
import numpy as np
from xpmir.rankers import AbstractModuleScorer


class DistillationListwiseLoss(Config, nn.Module):
    """The abstract loss for pairwise distillation"""

    weight: Param[float] = 1.0
    NAME = "?"

    def initialize(self, ranker: AbstractModuleScorer):
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


class DistillationListwiseTrainer(LossTrainer):
    """Listwise trainer for distillation"""

    sampler: Param[DistillationListwiseSampler] = field(overrides=True)
    """The sampler"""

    lossfn: Param[DistillationListwiseLoss]
    """The distillation pairwise batch function"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(self.model)
        for loss in context.hooks(DistillationListwiseLoss):
            loss.initialize(self.model)
        
        self.sampler.initialize(random)
        self.sampler_iter = self.sampler.listwise_iter()

        self.sampler_iter = MultiprocessSerializableIterator(
            self.sampler.listwise_batch_iter(self.batch_size)
        )

    def train_batch(self, samples: List[DistillationListwiseSampler]):
        # Builds records and teacher score matrix
        teacher_scores = torch.empty(len(samples), len(samples[0].documents))
        records = ListwiseRecords()
        for ix, sample in enumerate(samples):
            records.add(
                ListwiseRecord(
                    sample.query,
                    [doc.document for doc in sample.documents],
                )
            )
            teacher_scores[ix] = torch.tensor([doc.score for doc in sample.documents])

        # Get the next batch and compute the scores for each query/document
        #TODO debug : out should be of shape [2* len(records)], not the case for now
        scores = self.model(records, self.context).reshape(2, len(records)).T

        if torch.isnan(scores).any() or torch.isinf(scores).any():
            self.logger.error(
                "nan or inf relevance score detected. Aborting (pairwise distillation)."
            )
            sys.exit(1)

        # Call the losses (distillation, pairwise and pointwise)
        teacher_scores = teacher_scores.to(scores.device)
        self.lossfn.process(scores, teacher_scores, self.context)
