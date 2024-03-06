from abc import ABC, abstractmethod
from typing import List
from datamaestro.record import Record
from datamaestro_text.data.conversation import (
    DecontextualizedItem,
)
from xpmir.text.encoders import TextEncoderBase
from xpmir.letor.trainers.alignment import RepresentationOutput
from xpmir.utils.convert import Converter


class ConversationRepresentationEncoder(
    TextEncoderBase[List[Record], RepresentationOutput], ABC
):
    @abstractmethod
    def forward(self, conversations: List[Record]) -> RepresentationOutput:
        """Represents a list of conversations"""
        ...


class DecontextualizedQueryConverter(Converter[Record, str]):
    def __call__(self, input: Record) -> str:
        return input[DecontextualizedItem].get_decontextualized_query()
