from abc import ABC, abstractmethod
from attrs import define
import torch
from typing import List
from experimaestro import Config, Param
from datamaestro_text.data.conversation import Conversation
from xpmir.text.encoders import TextEncoder, TextEncoderBase
from xpmir.letor.trainers.alignment import RepresentationOutput


class ConversationRepresentationEncoder(
    TextEncoderBase[List[Conversation], RepresentationOutput], ABC
):
    @abstractmethod
    def forward(self, conversations: List[Conversation]) -> RepresentationOutput:
        """Represents a list of conversations"""
        ...


class GoldQueryConversationRepresentationEncoder(ConversationRepresentationEncoder):
    encoder: Param[TextEncoder]

    def forward(self, conversations: List[Conversation]) -> RepresentationOutput:
        texts = [conversation.decontextualized_query for conversation in conversations]
        return RepresentationOutput(self.encoder(texts))


class QueryRewritingSamplerLoss(Config):
    weight: Param[float] = 1.0
    """The weight for this loss"""


@define
class ConversationRepresentationOutput:
    representation: torch.Tensor


class ContextualizedRepresentationLoss(Config):
    def __call__(
        self,
        input: ConversationRepresentationOutput,
        target: ConversationRepresentationOutput,
    ):
        pass


class MSEContextualizedRepresentationLoss(ContextualizedRepresentationLoss):
    """Computes the asymetric loss for CoSPLADE"""

    def __call__(
        self,
        input: ConversationRepresentationOutput,
        target: ConversationRepresentationOutput,
    ):
        return torch.nn.functional.mse_loss(
            target.representation, input.representation, 0
        )
