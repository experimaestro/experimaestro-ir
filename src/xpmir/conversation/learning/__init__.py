from abc import ABC, abstractmethod
from functools import cached_property
from typing import Iterator

import numpy as np
from datamaestro_text.data.conversation import Conversation, ConversationDataset
from experimaestro import Param

from xpmir.conversation.records import HistoryRecord
from xpmir.learning.base import BaseSampler, Sampler
from xpmir.utils.iter import RandomSerializableIterator, SerializableIterator


class ConversationSampler(Sampler, ABC):
    @abstractmethod
    def iter(self) -> SerializableIterator[Conversation]:
        pass


class DatasetConversationSampler(Sampler):
    """Sampler for a contextualized query rewriting datasets"""

    dataset: Param[ConversationDataset]
    """The dataset used by the sampler"""

    @cached_property
    def data(self):
        return [x for x in self.dataset.iter()]

    def iter(self) -> RandomSerializableIterator[Conversation]:
        def generator(random):
            while True:
                yield self.data[random.randint(0, len(self.data))]

        return RandomSerializableIterator(self.random, generator)


class DatasetConversationEntrySampler(BaseSampler):
    """Uses a conversation dataset and sample entries from them"""

    dataset: Param[ConversationDataset]

    @cached_property
    def conversations(self):
        return list(self.dataset.iter_conversations())

    def __iter__(self) -> Iterator[HistoryRecord]:
        def generator(random: np.random.RandomState):
            while True:
                conversation_ix = random.randint(0, len(self.conversations))
                conversation = self.conversations[conversation_ix]
                entry_ix = random.randint(len(conversation.history))
                yield HistoryRecord(
                    conversation.history[entry_ix].topic,
                    conversation.history[:entry_ix],
                )

        return RandomSerializableIterator(self.random, generator)
