import torch
import itertools
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)


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


class BaseRecords:
    """Base records just exposes iterables on (query, document) pairs

    Records can be structured, i.e. the same queries and documents
    can be used more than once. To allow optimization (e.g. pre-computing
    document/query representation),
    """

    @property
    def queries(self) -> List[Query]:
        """Iterates over queries"""
        raise NotImplementedError(f"queries() in {self.__class__}")

    @property
    def unique_queries(self) -> Iterable[Query]:
        return self.queries

    @property
    def documents(self) -> Iterable[Document]:
        """Iterates over documents"""
        raise NotImplementedError(f"queries() in {self.__class__}")

    @property
    def unique_documents(self) -> Iterable[Document]:
        return self.documents

    def pairs(self) -> Tuple[List[int], List[int]]:
        """Returns the list of query/document indices for which we should compute the score,
        or None if all (cartesian product). This method should be used with `unique` set
        to true to get the queries/documents"""
        raise NotImplementedError(f"masks() in {self.__class__}")

    def __getitem__(self, ix: Union[slice, int]):
        """Sub-sample"""
        raise NotImplementedError(f"__getitem__() in {self.__class__}")


class PointwiseRecords(BaseRecords):
    """Pointwise records are a set of triples (query, document, relevance)"""

    # The queries
    queries: List[Query]

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
    """A pairwise record is composed of a query, a positive and a negative document"""

    query: Query
    positive: Document
    negative: Document

    def __init__(self, query: Query, positive: Document, negative: Document):
        self.query = query
        self.positive = positive
        self.negative = negative


class PairwiseRecords(BaseRecords):
    """Pairwise records of queries associated with (positive, negative) pairs"""

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
    def unique_queries(self):
        return self._queries

    @property
    def documents(self):
        return itertools.chain(self.positives, self.negatives)

    def pairs(self):
        indices = list(range(len(self._queries)))
        return indices * 2, list(range(2 * len(self.positives)))

    def __len__(self):
        return len(self._queries)

    def __getitem__(self, ix: Union[slice, int]):
        if isinstance(ix, slice):
            records = PairwiseRecords()
            for i in range(ix.start, min(ix.stop, len(self._queries)), ix.step or 1):
                records.add(
                    PairwiseRecord(
                        self._queries[i], self.positives[i], self.negatives[i]
                    )
                )
            return records

        return PairwiseRecord(self._queries[ix], self.positives[ix], self.negatives[ix])


class BatchwiseRecords(BaseRecords):
    """Several documents (with associated [pseudo]relevance) per query

    Assumes that the number of documents per query is always the same (even
    though documents themselves can be different)
    """

    relevances: torch.Tensor


class ProductRecords(BatchwiseRecords):
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
    def unique_queries(self):
        return self._queries

    @property
    def documents(self):
        for _ in self._queries:
            for d in self._documents:
                yield d

    @property
    def unique_documents(self):
        return self._documents

    def pairs(self):
        return None
