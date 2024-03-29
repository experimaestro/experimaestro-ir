from functools import cached_property
from datamaestro.record import Record
import numpy as np
from datamaestro_text.data.conversation import (
    ConversationDataset,
    ConversationHistoryItem,
    EntryType,
)
from experimaestro import Param

from xpmir.learning.base import BaseSampler
from xpmir.utils.iter import RandomSerializableIterator


class DatasetConversationEntrySampler(BaseSampler):
    """Uses a conversation dataset and topic records entries"""

    dataset: Param[ConversationDataset]
    """The conversation dataset"""

    @cached_property
    def conversations(self):
        return list(self.dataset.__iter__())

    def __post_init__(self):
        super().__post_init__()

    def __iter__(self) -> RandomSerializableIterator[Record]:
        def generator(random: np.random.RandomState):
            while True:
                # Pick a random conversation
                conversation_ix = random.randint(0, len(self.conversations))
                conversation = self.conversations[conversation_ix]

                # Pick a random topic record entry
                nodes = [
                    node
                    for node in conversation
                    if node.entry()[EntryType] == EntryType.USER_QUERY
                ]
                node_ix = random.randint(len(nodes))
                node = nodes[node_ix]

                node = node.entry().update(ConversationHistoryItem(node.history()))

                yield node

        return RandomSerializableIterator(self.random, generator)
