from typing import List, Optional

import numpy as np

from datamaestro_text.data.ir import DocumentStore

from xpmir.letor.records import MaskedLanguageModelingRecord
from xpmir.letor.samplers import Sampler
from experimaestro import Param

from xpmir.utils.iter import RandomSerializableIterator
from xpmir.utils.utils import easylog
import logging

logger = easylog()

logger = logging.getLogger()
logger.setLevel(logging.INFO)


class MLMSampler(Sampler):
    """Sample texts from various sources

    This sampler can be used for Masked Language Modeling to sample from several
    datasets.
    """

    datasets: Param[List[DocumentStore]]
    """Lists of datasets to sample from"""

    _stores: List[DocumentStore]

    def initialize(self, random: Optional[np.random.RandomState]):
        super().initialize(random)

    def record_iter(self) -> RandomSerializableIterator[MaskedLanguageModelingRecord]:
        def iter(random: np.random.RandomState):
            while True:
                # We could imagine setting weights here to give more importance
                # to one dataset
                document_count = [dataset.documentcount for dataset in self.datasets]
                choice = random.randint(0, len(self.datasets))
                if document_count[choice] < 10_000_000:
                    document = self.datasets[choice].document_int(
                        self.random.randint(0, self.datasets[choice].documentcount)
                    )
                    yield MaskedLanguageModelingRecord(
                        document.get_id(), document.get_text()
                    )
                else:
                    # FIXME: it makes the iter not fully serilizable
                    yield from (
                        MaskedLanguageModelingRecord(doc.get_id(), doc.get_text())
                        for doc in self.datasets[choice].iter()
                    )

        return RandomSerializableIterator(self.random, iter)
