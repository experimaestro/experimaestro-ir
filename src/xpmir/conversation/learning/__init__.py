import numpy as np
from datamaestro_text.data.ir import TopicRecord
from datamaestro_text.data.conversation import (
    ConversationDataset,
    ConversationTopicRecord,
)
from experimaestro import Param

from xpmir.learning.base import BaseSampler
from xpmir.utils.iter import RandomSerializableIterator


class DatasetConversationEntrySampler(BaseSampler):
    """Uses a conversation dataset and topic records entries"""

    dataset: Param[ConversationDataset]
    """The conversation dataset"""

    def __iter__(self) -> RandomSerializableIterator[ConversationTopicRecord]:
        def generator(random: np.random.RandomState):
            while True:
                # Pick a random conversation
                conversation_ix = random.randint(0, len(self.dataset))
                conversation = self.dataset[conversation_ix]

                # Pick a random topic record entry
                nodes = [
                    node
                    for node in conversation
                    if isinstance(node.entry(), TopicRecord)
                ]
                node_ix = random.randint(len(nodes))
                node = nodes[node_ix]

                yield ConversationTopicRecord(node.entry(), node.history())

        return RandomSerializableIterator(self.random, generator)
