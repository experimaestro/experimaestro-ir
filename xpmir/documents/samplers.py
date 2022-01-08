from typing import Tuple, Iterator
from experimaestro import Config, Param

from datamaestro_text.data.ir import AdhocDocumentStore


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
