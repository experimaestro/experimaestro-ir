from typing import Tuple, Iterator
from experimaestro import Config, Param
import torch
import re
import numpy as np
from datamaestro_text.data.ir import AdhocDocumentStore
from xpmir.letor.records import Document, ProductRecords, Query
from xpmir.letor.samplers import (
    BatchwiseSampler,
    RandomSerializableIterator,
    SerializableIterator,
    BatchwiseRecords,
)


class DocumentSampler(Config):
    """How to sample from a document store"""

    documents: Param[AdhocDocumentStore]

    def __call__(self) -> Tuple[int, Iterator[str]]:
        raise NotImplementedError()


class HeadDocumentSampler(DocumentSampler):
    """A basic sampler that iterates over the first documents"""

    max_count: Param[int] = 0
    """Maximum number of documents (if 0, no limit)"""

    max_ratio: Param[float] = 0
    """Maximum ratio of documents (if 0, no limit)"""

    def __call__(self) -> Tuple[int, Iterator[str]]:
        count = (self.max_ratio or 1) * self.documents.documentcount

        if self.max_count > 0:
            count = min(self.max_count, count)

        count = int(count)
        return count, self.iter(count)

    def iter(self, count):
        for ix, document in zip(range(count), self.documents.iter_documents()):
            yield document.text


class BatchwiseRandomSpanSampler(DocumentSampler, BatchwiseSampler):
    """This sampler uses positive samples coming from the same documents and negative ones coming from others

    Allows to (pre)-train as in co-condenser:
        L. Gao and J. Callan, “Unsupervised Corpus Aware Language Model Pre-training for Dense Passage Retrieval,” arXiv:2108.05540 [cs], Aug. 2021, Accessed: Sep. 17, 2021. [Online]. Available: http://arxiv.org/abs/2108.05540
    """

    max_spansize: Param[int] = 1000
    """Maximum span size in number of characters"""

    def batchwise_iter(self, batch_size: int) -> SerializableIterator[ProductRecords]:
        def iterator(random: np.random.RandomState):
            # Pre-compute relevance matrix
            relevances = torch.diag(torch.ones(batch_size, dtype=torch.float))

            iter = self.documents.iter_sample(lambda m: random.randint(0, m))

            while True:
                batch = ProductRecords()
                while len(batch) < batch_size:
                    record = next(iter)
                    text = record.text
                    spanlen = min(self.max_spansize, len(text) // 2)

                    max_start1 = len(text) - spanlen * 2
                    start1 = random.randint(0, max_start1) if max_start1 > 0 else 0
                    end1 = start1 + spanlen
                    if start1 > 0 and text[start1 - 1] != " ":
                        start1 = text.find(" ", start1) + 1
                    if text[end1] != " ":
                        end1 = text.rfind(" ", 0, end1)

                    max_start2 = len(text) - spanlen
                    start2 = (
                        random.randint(end1, max_start2) if max_start2 > end1 else end1
                    )
                    end2 = start2 + spanlen
                    if text[start2 - 1] != " ":
                        start2 = text.find(" ", start2) + 1
                    if text[end2 - 1] != " ":
                        end2 = text.rfind(" ", 0, end2)

                    # Rejet wrong samples
                    if end2 <= start2 or end1 <= start1:
                        continue

                    batch.addQueries(Query(None, text[start1:end1]))
                    batch.addDocuments(Document(None, text[start2:end2], 0))
                batch.setRelevances(relevances)
                yield batch

        return RandomSerializableIterator(self.random, iterator)
