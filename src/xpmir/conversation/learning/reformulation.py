from abc import ABC, abstractmethod
from typing import List
from experimaestro import Config, Param
from datamaestro_text.data.conversation import Conversation
from xpmir.learning import Module
from xpmir.text.encoders import TextEncoder
from xpmir.letor.trainers.alignment import RepresentationOutput


class ConversationRepresentationEncoder(Module, ABC):
    @abstractmethod
    def forward(self, conversations: List[Conversation]) -> RepresentationOutput:
        pass


class GoldQueryConversationRepresentationEncoder(ConversationRepresentationEncoder):
    encoder: Param[TextEncoder]

    def forward(self, conversations: List[Conversation]) -> RepresentationOutput:
        texts = [conversation.decontextualized_query for conversation in conversations]
        return RepresentationOutput(self.encoder(texts))


class QueryRewritingSamplerLoss(Config):
    weight: Param[float] = 1.0
    """The weight for this loss"""
