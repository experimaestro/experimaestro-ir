from functools import cached_property
from typing import Iterator, List

import numpy as np
from datamaestro.record import Record
from datamaestro_text.data.conversation import (
    ConversationDataset,
    ConversationHistoryItem,
    EntryType,
)
from experimaestro import Config, Param

from xpmir.learning.base import BaseSampler, SampleIterator
from xpmir.utils.iter import RandomSerializableIterator


class DatasetConversationBase(Config):
    datasets: Param[List[ConversationDataset]]
    """The conversation datasets"""

    @cached_property
    def records(self):
        records = []
        for dataset in self.datasets:
            for conversation in dataset.__iter__():
                nodes = [
                    node
                    for node in conversation
                    if node.entry[EntryType] == EntryType.USER_QUERY
                ]
                for node in nodes:
                    records.append(
                        node.entry.update(ConversationHistoryItem(node.history()))
                    )

        return records


class DatasetConversationIterator(SampleIterator, DatasetConversationBase):
    def __iter__(self) -> Iterator[Record]:
        yield from self.records


class DatasetConversationEntrySampler(BaseSampler, DatasetConversationBase):
    """Uses a conversation dataset and topic records entries"""

    def __iter__(self) -> RandomSerializableIterator[Record]:
        def generator(random: np.random.RandomState):
            while True:
                yield self.records[random.randint(0, len(self.records))]

        return RandomSerializableIterator(self.random, generator)
