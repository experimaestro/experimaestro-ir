from abc import ABC, abstractmethod
from dataclasses import InitVar
import sys
from typing import Dict, Iterator, List
from attr import define
import torch
from experimaestro import Config, Param
from datamaestro_text.data.conversation import Conversation
from xpmir.learning import Module
from xpmir.learning.context import Loss
from xpmir.text.encoders import TextEncoder
from xpmir.letor.trainers import TrainerContext, LossTrainer
import numpy as np
from xpmir.utils.iter import (
    RandomSerializableIterator,
    SerializableIterator,
)


@define
class ConversationRepresentationOutput:
    representation: torch.Tensor


class ConversationRepresentationEncoder(Module, ABC):
    @abstractmethod
    def forward(
        self, conversations: List[Conversation]
    ) -> ConversationRepresentationOutput:
        pass


class GoldQueryConversationRepresentationEncoder(ConversationRepresentationEncoder):
    encoder: Param[TextEncoder]

    def forward(
        self, conversations: List[Conversation]
    ) -> ConversationRepresentationOutput:
        texts = [conversation.decontextualized_query for conversation in conversations]
        return ConversationRepresentationOutput(self.encoder(texts))


class QueryRewritingSamplerLoss(Config):
    weight: Param[float] = 1.0
    """The weight for this loss"""


class ReformulationTrainerBase(LossTrainer):
    """Base reformulation-based trainer"""

    losses: Param[Dict[str, QueryRewritingSamplerLoss]]
    """The loss function(s)"""

    sampler: Param[ContextualizedQueryRewritingSamplerBase]
    """The pairwise sampler"""

    sampler_iter: InitVar[SerializableIterator[Conversation]]
    """The iterator over samples"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.sampler.initialize(random)
        self.sampler_iter = self.sampler.iter()

    def iter_batches(self) -> Iterator[Conversation]:
        while True:
            batch = Conversation()
            for _, record in zip(range(self.batch_size), self.sampler_iter):
                batch.add(record)
            yield batch


class ContextualizedRepresentationLoss(QueryRewritingSamplerLoss, ABC):
    @abstractmethod
    def __call__(
        self,
        input: ConversationRepresentationOutput,
        target: ConversationRepresentationOutput,
    ):
        """Computes the reconstruction loss

        :param target: a tensor of size BxD
        """
        ...


class MSEContextualizedRepresentationLoss(ContextualizedRepresentationLoss):
    """Computes the MSE between contextualized query representation and gold
    representation"""

    def __post_init__(self):
        self.mse = torch.nn.MSELoss()

    def __call__(
        self,
        input: ConversationRepresentationOutput,
        target: ConversationRepresentationOutput,
    ):
        return self.mse(input.representation, target.representation)


class RepresentationReformulationTrainer(ReformulationTrainerBase):
    """Compares the contextualized query representation with an expected query
    representation

    Both the representations are expected to a be in a vector space
    """

    losses: Param[Dict[str, ContextualizedRepresentationLoss]]
    """The loss function"""

    target_model: Param[ConversationRepresentationEncoder]
    """Target model"""

    def train_batch(self, records: List[Conversation]):
        # Get the next batch and compute the scores for each query/document
        output: ConversationRepresentationOutput = self.model(records, self.context)
        target: ConversationRepresentationOutput = self.target_model(records)

        if (
            torch.isnan(output.representation).any()
            or torch.isinf(output.representation).any()
        ):
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        for key, loss in self.losses.items():
            value = loss(output, target)
            self.context.add_loss(Loss(key, value, loss.weight))
