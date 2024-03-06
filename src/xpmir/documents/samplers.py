from abc import ABC, abstractmethod
from typing import Optional, Tuple, Iterator, Any
from experimaestro import Param, Config
import torch
import numpy as np
from datamaestro_text.data.ir import DocumentStore, TextItem, create_record
from xpmir.letor import Random
from xpmir.letor.records import DocumentRecord, PairwiseRecord, ProductRecords
from xpmir.letor.samplers import BatchwiseSampler, PairwiseSampler
from xpmir.utils.iter import RandomSerializableIterator, SerializableIterator


class DocumentSampler(Config, ABC):
    """How to sample from a document store"""

    documents: Param[DocumentStore]

    @abstractmethod
    def __call__(self) -> Tuple[Optional[int], Iterator[DocumentRecord]]:
        """Returns an indicative number of samples and an iterator"""
        raise NotImplementedError()

    def __iter__(self) -> Iterator[DocumentRecord]:
        """Shorthand method that directly returns an iterator"""
        _, iter = self()
        return iter


class HeadDocumentSampler(DocumentSampler):
    """A basic sampler that iterates over the first documents

    if max_count is 0, it iterates over all documents
    """

    max_count: Param[int] = 0
    """Maximum number of documents (if 0, no limit)"""

    max_ratio: Param[float] = 0
    """Maximum ratio of documents (if 0, no limit)"""

    def __call__(self) -> Tuple[int, Iterator[DocumentRecord]]:
        count = (self.max_ratio or 1) * self.documents.documentcount

        if self.max_count > 0:
            count = min(self.max_count, count)

        count = int(count)
        return count, self.iter(count)

    def iter(self, count):
        for _, document in zip(range(count), self.documents.iter_documents()):
            yield document


class RandomDocumentSampler(DocumentSampler):
    """A basic sampler that iterates over the first documents

    Either max_count or max_ratio should be non null
    """

    max_count: Param[int] = 0
    """Maximum number of documents (if 0, no limit)"""

    max_ratio: Param[float] = 0
    """Maximum ratio of documents (if 0, no limit)"""

    random: Param[Optional[Random]]
    """Random sampler"""

    def __call__(self) -> Tuple[int, Iterator[str]]:
        # Compute the number of documents to sample
        count = (self.max_ratio or 1) * self.documents.documentcount

        if self.max_count > 0:
            count = min(self.max_count, count)
        count = int(count)
        return count, self.iter(count)

    def iter(self, count) -> Iterator[str]:
        """Iterate over the documents"""
        state = np.random.RandomState() if self.random is None else self.random.state
        docids = state.choice(
            np.arange(self.documents.documentcount), size=count, replace=False
        )
        for docid in docids:
            yield self.documents.document_int(int(docid))


class RandomSpanSampler(BatchwiseSampler, PairwiseSampler):
    """This sampler uses positive samples coming from the same documents
    and negative ones coming from others

    Allows to (pre)-train as in co-condenser:
        L. Gao and J. Callan, “Unsupervised Corpus Aware Language Model
        Pre-training for Dense Passage Retrieval,” arXiv:2108.05540 [cs],
        Aug. 2021, Accessed: Sep. 17, 2021. [Online].
        http://arxiv.org/abs/2108.05540
    """

    documents: Param[DocumentStore]
    """The document store to use"""

    max_spansize: Param[int] = 1000
    """Maximum span size in number of characters"""

    def get_text_span(self, text, random):
        # return the two spans of text
        spanlen = min(self.max_spansize, len(text) // 2)

        max_start1 = len(text) - spanlen * 2
        start1 = random.randint(0, max_start1) if max_start1 > 0 else 0
        end1 = start1 + spanlen
        if start1 > 0 and text[start1 - 1] != " ":
            start1 = text.find(" ", start1) + 1
        if text[end1] != " ":
            end1 = text.rfind(" ", 0, end1)

        max_start2 = len(text) - spanlen
        start2 = random.randint(end1, max_start2) if max_start2 > end1 else end1
        end2 = start2 + spanlen
        if text[start2 - 1] != " ":
            start2 = text.find(" ", start2) + 1
        if text[end2 - 1] != " ":
            end2 = text.rfind(" ", 0, end2)

        # Rejet wrong samples
        if end2 <= start2 or end1 <= start1:
            return None

        return (text[start1:end1], text[start2:end2])

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord, Any]:
        def iter(random: np.random.RandomState):
            iter = self.documents.iter_sample(lambda m: random.randint(0, m))

            while True:
                record_pos_qry = next(iter)
                text_pos_qry = record_pos_qry[TextItem].text
                spans_pos_qry = self.get_text_span(text_pos_qry, random)

                record_neg = next(iter)
                text_neg = record_neg[TextItem].text
                spans_neg = self.get_text_span(text_neg, random)

                if not (spans_pos_qry and spans_neg):
                    continue

                yield PairwiseRecord(
                    create_record(text=spans_pos_qry[0]),
                    create_record(text=spans_pos_qry[1]),
                    create_record(text=spans_neg[random.randint(0, 2)]),
                )

        return RandomSerializableIterator(self.random, iter)

    def batchwise_iter(
        self, batch_size: int
    ) -> SerializableIterator[ProductRecords, Any]:
        def iterator(random: np.random.RandomState):
            # Pre-compute relevance matrix
            relevances = torch.diag(torch.ones(batch_size, dtype=torch.float))

            iter = self.documents.iter_sample(lambda m: random.randint(0, m))

            while True:
                batch = ProductRecords()
                while len(batch) < batch_size:
                    record = next(iter)
                    text = record.text
                    res = self.get_text_span(text, random)
                    if not res:
                        continue
                    batch.add_topics(create_record(text=res[0]))
                    batch.add_documents(create_record(text=res[1]))
                batch.set_relevances(relevances)
                yield batch

        return RandomSerializableIterator(self.random, iterator)
