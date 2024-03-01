from abc import ABC, abstractmethod
from typing import List
from datamaestro_text.data.conversation import (
    TopicConversationRecord,
    DecontextualizedItem,
)
from xpmir.text.encoders import TextEncoderBase
from xpmir.letor.trainers.alignment import RepresentationOutput
from xpmir.utils.convert import Converter


class ConversationRepresentationEncoder(
    TextEncoderBase[List[TopicConversationRecord], RepresentationOutput], ABC
):
    @abstractmethod
    def forward(
        self, conversations: List[TopicConversationRecord]
    ) -> RepresentationOutput:
        """Represents a list of conversations"""
        ...


class DecontextualizedQueryConverter(Converter[TopicConversationRecord, str]):
    def __call__(self, input: TopicConversationRecord) -> str:
        if isinstance(input, TopicConversationRecord):
            input = input.record
        return input[DecontextualizedItem].get_decontextualized_query()
