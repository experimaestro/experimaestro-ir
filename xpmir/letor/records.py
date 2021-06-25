import torch
import itertools
from typing import List, Optional


class Document:
    docid: str
    text: str
    score: float

    def __init__(self, docid, text, score):
        self.docid = docid
        self.text = text
        self.score = score


class PointwiseRecord:
    """A record from a pointwise sampler"""

    # The query
    query: str

    # The document
    document: Document

    # The relevance
    relevance: Optional[float]

    def __init__(self, query, docid, document, score, relevance):
        self.query = query
        self.document = Document(docid, document, score)
        self.relevance = relevance


class TokenizedTexts:
    def __init__(
        self,
        tokens: List[List[str]],
        ids: torch.LongTensor,
        lens: List[int],
        mask: torch.LongTensor,
    ):
        self.tokens = tokens
        self.ids = ids
        self.lens = lens
        self.mask = mask


class Records:
    """Records are the objects passed to the module forwards"""

    # The queries
    queries: List[str]

    # Text of the documents
    documents: List[Document]

    # The relevances
    relevances: List[float]

    def __init__(self):
        self.queries = []
        self.documents = []
        self.relevances = []

    def add(self, record: PointwiseRecord):
        self.queries.append(record.query)
        self.relevances.append(record.relevance or 0)
        self.documents.append(record.document)


class PairwiseRecord:
    query: str
    positive: Document
    negative: Document

    def __init__(self, query: str, positive: Document, negative: Document):
        self.query = query
        self.positive = positive
        self.negative = negative


class PairwiseRecords:
    """"""

    # The queries
    _queries: List[str]

    # The document IDs (positive)
    positives: List[Document]

    # The scores of the retriever
    negatives: List[Document]

    def __init__(self):
        self._queries = []
        self.positives = []
        self.negatives = []

    def add(self, record: PairwiseRecord):
        self._queries.append(record.query)
        self.positives.append(record.positive)
        self.negatives.append(record.negative)

    @property
    def queries(self):
        return itertools.chain(self._queries, self._queries)

    @property
    def documents(self):
        return itertools.chain(self.positives, self.negatives)
