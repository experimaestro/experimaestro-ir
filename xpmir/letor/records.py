import torch
import itertools
from typing import Iterable, List, NamedTuple, Optional, Tuple


class Query(NamedTuple):
    text: str


class Document(NamedTuple):
    docid: str
    text: str
    score: float


class PointwiseRecord:
    """A record from a pointwise sampler"""

    # The query
    query: Query

    # The document
    document: Document

    # The relevance
    relevance: Optional[float]

    def __init__(
        self,
        query: Query,
        docid: str,
        content: str,
        score: float,
        relevance: Optional[float],
    ):
        self.query = query
        self.document = Document(docid, content, score)
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


# The mask is a list (queries) of list (document index)
QDMask = List[List[int]]

# Structured iterator for a batch
StructuredIterator = Iterable[Tuple[List[Query], List[Document], Optional[QDMask]]]


class BaseRecords:
    """Base records just exposes iterables on (query, document) pairs"""

    queries: Iterable[Query]
    documents: Iterable[Document]

    def structured(self) -> StructuredIterator:
        """Returns structured query/document couples"""
        for q, d in zip(self.queries, self.documents):
            yield ([q], [d], None)


class PointwiseRecords(BaseRecords):
    """Pointwise records are the objects passed to the module forwards"""

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
    query: Query
    positive: Document
    negative: Document

    def __init__(self, query: Query, positive: Document, negative: Document):
        self.query = query
        self.positive = positive
        self.negative = negative


class PairwiseRecords(BaseRecords):
    """Pairwise records with (positive, negative) pairs"""

    # The queries
    _queries: List[Query]

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

    def structured(self) -> StructuredIterator:
        for q, p, n in zip(self._queries, self.positives, self.negatives):
            yield ([q], [p, n], None)


class BatchwiseRecords(BaseRecords):
    """Several documents (with associated [pseudo]relevance) per query

    Assumes that the number of documents per query is always the same (even
    though documents themselves can be different)
    """

    relevances: torch.Tensor


class CartesianProductRecords(BatchwiseRecords):
    """Computes the score for all the documents and queries

    Attributes:

        _queries: The list of queries
        _documents: The list of documents
        _relevances: (query x document) matrix with relevance score (between 0 and 1)
    """

    _queries: List[Query]
    _documents: List[Document]

    def __init__(self):
        self._queries = []
        self._documents = []

    def addQueries(self, *queries: Query):
        self._queries.extend(queries)

    def addDocuments(self, *documents: Document):
        self._documents.extend(documents)

    def setRelevances(self, relevances: torch.Tensor):
        assert relevances.shape[0] == len(self._queries)
        assert relevances.shape[1] == len(self._documents)
        self.relevances = relevances

    @property
    def queries(self):
        for q in self._queries:
            for _ in self._documents:
                yield q

    @property
    def documents(self):
        for _ in self._queries:
            for d in self._documents:
                yield d

    def structured(self) -> StructuredIterator:
        return [(self._queries, self._documents, None)]
