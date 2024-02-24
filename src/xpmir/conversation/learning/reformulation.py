from abc import ABC, abstractmethod
from typing import List
from datamaestro_text.data.conversation import (
    ConversationTopicRecord,
    DecontextualizedRecord,
)
from xpmir.text.encoders import TextEncoderBase
from xpmir.letor.trainers.alignment import RepresentationOutput
from xpmir.utils.convert import Converter


class ConversationRepresentationEncoder(
    TextEncoderBase[List[ConversationTopicRecord], RepresentationOutput], ABC
):
    @abstractmethod
    def forward(
        self, conversations: List[ConversationTopicRecord]
    ) -> RepresentationOutput:
        """Represents a list of conversations"""
        ...


class DecontextualizedQueryConverter(Converter[DecontextualizedRecord, str]):
    def __call__(self, input: DecontextualizedRecord) -> str:
        if isinstance(input, ConversationTopicRecord):
            input = input.record
        return input.get_decontextualized_query()
