from abc import ABC, abstractmethod
from dataclasses import InitVar
from functools import cached_property
import math
import sys
from typing import Iterator
import torch
from torch import nn
from torch.functional import Tensor
import torch.nn.functional as F
from experimaestro import Config, Param
from datamaestro_text.data.conversation import (
    ContextualizedQueryRewriting,
    ContextualizedRewrittenQuery,
)
from xpmir.learning.context import Loss
from xpmir.learning.metrics import ScalarMetric
from xpmir.learning.base import Sampler
from xpmir.letor.trainers import TrainerContext, LossTrainer
import numpy as np
from xpmir.rankers import LearnableScorer, ScorerOutputType
from xpmir.utils.utils import foreach
from xpmir.utils.iter import (
    RandomSerializableIterator,
    SerializableIterator,
)


class ContextualizedQueryRewritingSamplerBase(Sampler, ABC):
    @abstractmethod
    def iter(self) -> SerializableIterator[ContextualizedRewrittenQuery]:
        pass


class ContextualizedQueryRewritingSampler(Sampler):
    """Sampler for a contextualized query rewriting datasets"""

    dataset: Param[ContextualizedQueryRewriting]
    """The dataset used by the sampler"""

    @cached_property
    def data(self):
        return [x for x in self.dataset.iter()]

    def iter(self) -> RandomSerializableIterator[ContextualizedRewrittenQuery]:
        def generator(random):
            while True:
                yield self.data[random.randint(0, len(self.data))]

        return RandomSerializableIterator(self.random, generator)


class QueryRewritingSamplerLoss(Config):
    pass


class ReformulationTrainerBase(LossTrainer):
    """Base reformulation-based trainer"""

    lossfn: Param[QueryRewritingSamplerLoss]
    """The loss function"""

    sampler: Param[ContextualizedQueryRewritingSamplerBase]
    """The pairwise sampler"""

    sampler_iter: InitVar[SerializableIterator[ContextualizedRewrittenQuery]]
    """The iterator over samples"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.sampler.initialize(random)
        self.sampler_iter = self.sampler.iter()

    def iter_batches(self) -> Iterator[ContextualizedRewrittenQuery]:
        while True:
            batch = ContextualizedRewrittenQuery()
            for _, record in zip(range(self.batch_size), self.sampler_iter):
                batch.add(record)
            yield batch


class ContextualizedRepresentationLoss(QueryRewritingSamplerLoss, ABC):
    @abstractmethod
    def __call__(self, input: torch.Tensor, target: torch.Tensor):
        """Computes the loss given two tensor of shape (B, d) where B is the
        batch size and d the dimension of the representation"""
        ...


class MSEContextualizedRepresentationLoss(ContextualizedRepresentationLoss):
    """Computes the MSE between contextualized query representation and gold
    representation"""

    def __post_init__(self):
        self.mse = torch.nn.MSELoss()

    def __call__(self, input: torch.Tensor, target: torch.Tensor):
        return self.mse(input, target)


class RepresentationReformulationTrainer(ReformulationTrainerBase):
    """Compares the contextualized query representation with an expected query representation"""

    lossfn: Param[ContextualizedRepresentationLoss]
    """The loss function"""

    def train_batch(self, records: ContextualizedRewrittenQuery):
        # Get the next batch and compute the scores for each query/document
        rel_scores = self.ranker(records, self.context)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        pairwise_scores = rel_scores.reshape(2, len(records)).T
        self.lossfn.process(pairwise_scores, self.context)
