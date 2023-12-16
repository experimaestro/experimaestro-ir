from abc import ABC, abstractmethod
from functools import cached_property
from datamaestro_text.data.conversation import Conversation, ConversationDataset

from experimaestro import Config, Param
from xpmir.learning.base import Sampler
from xpmir.utils.iter import (
    RandomSerializableIterator,
    SerializableIterator,
)


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
