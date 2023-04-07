from dataclasses import dataclass
import torch
import itertools
from typing import (
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TypeVar,
    Union,
)


@dataclass
class Query:
    id: Optional[str]
    text: Optional[str]


class Document(NamedTuple):
    """A document (the docid or the text can be None, but not both)"""

    docid: Optional[str]
    text: Optional[str]
    score: Optional[float]


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


class TokenizedTexts(NamedTuple):
    tokens: List[List[str]]
    ids: torch.LongTensor
    lens: List[int]
    mask: torch.LongTensor
    token_type_ids: torch.LongTensor = None


RT = TypeVar("RT")


class BaseRecords(List[RT]):
    """Base records just exposes iterables on (query, document) pairs

    Records can be structured, i.e. the same queries and documents
    can be used more than once. To allow optimization (e.g. pre-computing
    document/query representation),
    """

    queries: Iterable[Query]
    documents: Iterable[Document]
    is_product = False

    @property
    def unique_queries(self) -> Iterable[Query]:
        return self.queries

    @property
    def unique_documents(self) -> Iterable[Document]:
        return self.documents

    def pairs(self) -> Tuple[Iterable[int], Iterable[int]]:
        """Returns two iterators (over queries and documents)

        Returns the list of query/document indices for which we should compute
        the score, or None if all (cartesian product). This method should be
        used with `unique` set to true to get the queries/documents
        """
        raise NotImplementedError(f"pairs() in {self.__class__}")

    def __getitem__(self, ix: Union[slice, int]):
        """Sub-sample"""
        raise NotImplementedError(f"__getitem__() in {self.__class__}")

    def __len__(self):
        """Returns the number of records

        The length is dependant on the type of records, and is mainly used
        to divide the data into batches
        """
        raise NotImplementedError(f"__len__() in {self.__class__}")


class PointwiseRecords(BaseRecords[PointwiseRecord]):
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

    def __len__(self):
        return len(self.queries)

    def pairs(self) -> Tuple[List[int], List[int]]:
        ix = list(range(len(self.queries)))
        return (ix, ix)

    @staticmethod
    def from_texts(
        queries: List[str],
        documents: List[str],
        relevances: Optional[List[float]] = None,
    ):
        records = PointwiseRecords()
        records.queries = list(map(lambda t: Query(None, t), queries))
        records.documents = list(map(lambda t: Query(None, t), documents))
        records.relevances = relevances
        return records


class PairwiseRecord:
    """A pairwise record is composed of a query, a positive and a negative document"""

    query: Query
    positive: Document
    negative: Document

    def __init__(self, query: Query, positive: Document, negative: Document):
        self.query = query
        self.positive = positive
        self.negative = negative


class PairwiseRecordWithTarget(PairwiseRecord):
    """A pairwise record is composed of a query, a positive and a negative
    document, and the indetifier which says the one on the first is pos or
    neg"""

    target: int

    def __init__(
        self, query: Query, positive: Document, negative: Document, target: int
    ):
        super().__init__(query, positive, negative)
        self.target = target


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


class PairwiseRecordsWithTarget(PairwiseRecords):
    """Pairwise records associated with a label (saying which document is better)"""

    target: List[int]

    def __init__(self):
        super().__init__()
        self.target = []

    def add(self, record: PairwiseRecordWithTarget):
        self._queries.append(record.query)
        self.positives.append(record.positive)
        self.negatives.append(record.negative)
        self.target.append(record.target)

    def get_target(self):
        return self.target

    def __getitem__(self, ix: Union[slice, int]):
        if isinstance(ix, slice):
            records = PairwiseRecordsWithTarget()
            for i in range(ix.start, min(ix.stop, len(self._queries)), ix.step or 1):
                records.add(
                    PairwiseRecordWithTarget(
                        self._queries[i],
                        self.positives[i],
                        self.negatives[i],
                        self.target[i],
                    )
                )
            return records

        return PairwiseRecordWithTarget(
            self._queries[ix], self.positives[ix], self.negatives[ix], self.target[ix]
        )


class BatchwiseRecords(BaseRecords):
    """Several documents (with associated [pseudo]relevance) per query

    Assumes that the number of documents per query is always the same (even
    though documents themselves can be different)
    """

    relevances: torch.Tensor

    def __getitem__(self, ix: Union[slice, int]):
        """Sub-sample"""
        raise NotImplementedError(f"__getitem__() in {self.__class__}")


class ProductRecords(BatchwiseRecords):
    """Computes the score for all the documents and queries

    The relevance matrix

    Attributes:

        _queries: The list of queries
        _documents: The list of documents
        _relevances: (query x document) matrix with relevance score (between 0 and 1)
    """

    _queries: List[Query]
    """The list of queries to score"""

    _documents: List[Document]
    """The list of documents to score"""

    relevances: torch.Tensor
    """A 2D tensor (query x document) indicating the relevance of the each
    query/document pair"""

    is_product = True

    def __init__(self):
        self._queries = []
        self._documents = []

    def addQueries(self, *queries: Query):
        self._queries.extend(queries)

    def addDocuments(self, *documents: Document):
        self._documents.extend(documents)

    def setRelevances(self, relevances: torch.Tensor):
        assert relevances.shape[0] == len(self._queries), (
            f"The number of queries {len(self._queries)} "
            + "does not match the number of rows {relevances.shape[0]}"
        )
        assert relevances.shape[1] == len(self._documents), (
            f"The number of documents {len(self._documents)} "
            + "does not match the number of columns {relevances.shape[1]}"
        )
        self.relevances = relevances

    def __len__(self):
        return len(self._queries)

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

    def pairs(self) -> Tuple[Iterable[int], Iterable[int]]:
        queries = []
        documents = []
        for q in range(len(self._queries)):
            for d in range(len(self._documents)):
                queries.append(q)
                documents.append(d)
        return queries, documents
