import torch
import itertools
from datamaestro_text.data.ir import (
    IDTextRecord,
    TextRecord,
    SimpleTextItem,
)
from typing import (
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    TypedDict,
    Union,
)
from typing_extensions import ReadOnly


## TypeddDicts

class ScoreRecord(TypedDict):
    """A record with just a score"""

    score: ReadOnly[float]


class ScoreDocumentRecord(IDTextRecord, ScoreRecord):
    """A record with an ID, a text item and a score"""

    pass




## Dataclasses / generics

DocT = TypeVar("DocT")
DocT2 = TypeVar("DocT2")
QueryT = TypeVar("QueryT")
QueryT2 = TypeVar("QueryT2")


class PointwiseRecord(Generic[DocT, QueryT]):
    """A record from a pointwise sampler"""

    # The query
    topic: QueryT

    # The document
    document: DocT

    # The relevance
    relevance: Optional[float]

    def __init__(
        self,
        topic: QueryT,
        document: DocT,
        relevance: Optional[float] = None,
    ):
        self.topic = topic
        self.document = document
        self.relevance = relevance

    @property
    def query(self):
        return self.topic

    def get_queries(self) -> List[QueryT]:
        return [self.topic]

    def with_queries(self, qs: "List[QueryT2]") -> "PointwiseRecord[DocT, QueryT2]":
        return PointwiseRecord(qs[0], self.document, self.relevance)

    def get_documents(self) -> List[DocT]:
        return [self.document]

    def with_documents(self, ds: "List[DocT2]") -> "PointwiseRecord[DocT2, QueryT]":
        return PointwiseRecord(self.topic, ds[0], self.relevance)

RT = TypeVar("RT")

class BaseRecords(List[RT]):
    """Base records just exposes iterables on (query, document) pairs

    Records can be structured, i.e. the same queries and documents
    can be used more than once. To allow optimization (e.g. pre-computing
    document/query representation),
    """

    topics: Iterable[IDTextRecord]
    documents: Iterable[IDTextRecord]
    is_product = False

    def __repr__(self):
        return f"<{self.__class__.__name__}(count={len(self)})>"

    @property
    def unique_topics(self) -> List[IDTextRecord]:

        return list(self.topics)

    unique_queries = unique_topics

    @property
    def unique_documents(self) -> List[IDTextRecord]:
        return list(self.documents)

    @property
    def queries(self):
        """Deprecated: use topics"""
        return self.topics

    def to(self, device):
        """Moves the records to a device (e.g. for the relevance matrix in ProductRecords)
        Default implementation does nothing, but can be implemented by specific records that have tensors as attributes (e.g. ProductRecords)
        """
        return self
    
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
    topics: List[IDTextRecord]

    # Text of the documents
    documents: List[IDTextRecord]

    # The relevances
    relevances: List[float]

    def __init__(self):
        self.topics = []
        self.documents = []
        self.relevances = []

    def add(self, record: PointwiseRecord):
        self.queries.append(record.query)
        self.relevances.append(record.relevance or 0)
        self.documents.append(record.document)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, ix: Union[slice, int]):
        if isinstance(ix, slice):
            records = PointwiseRecords()
            for i in range(ix.start, min(ix.stop, len(self.topics)), ix.step or 1):
                records.add(
                    PointwiseRecord(
                        self.topics[i], self.documents[i], self.relevances[i]
                    )
                )
            return records

        return PointwiseRecord(self.topics[ix], self.documents[ix], self.relevances[ix])

    def pairs(self) -> Tuple[List[int], List[int]]:
        ix = list(range(len(self.queries)))
        return (ix, ix)

    @staticmethod
    def from_texts(
        topics: List[str],
        documents: List[str],
        relevances: Optional[List[float]] = None,
    ):
        records = PointwiseRecords()
        records.topics = list(
            map(lambda t: TextRecord(text_item=SimpleTextItem(t)), topics)
        )
        records.documents = list(
            map(lambda t: TextRecord(text_item=SimpleTextItem(t)), documents)
        )
        records.relevances = relevances
        return records


class PairwiseRecord(Generic[DocT, QueryT]):
    """A pairwise record is composed of a query, a positive and a negative document"""

    query: QueryT
    positive: DocT
    negative: DocT

    def __init__(self, query: QueryT, positive: DocT, negative: DocT):
        self.query = query
        self.positive = positive
        self.negative = negative

    def get_queries(self) -> List[QueryT]:
        return [self.query]

    def with_queries(self, qs: "List[QueryT2]") -> "PairwiseRecord[DocT, QueryT2]":
        return PairwiseRecord(qs[0], self.positive, self.negative)

    def get_documents(self) -> List[DocT]:
        return [self.positive, self.negative]

    def with_documents(self, ds: "List[DocT2]") -> "PairwiseRecord[DocT2, QueryT]":
        return PairwiseRecord(self.query, ds[0], ds[1])


class PairwiseRecordWithTarget(PairwiseRecord):
    """A pairwise record is composed of a query, a positive and a negative
    document, and the indetifier which says the one on the first is pos or
    neg"""

    target: int

    def __init__(
        self,
        query: IDTextRecord,
        positive: IDTextRecord,
        negative: IDTextRecord,
        target: int,
    ):
        super().__init__(query, positive, negative)
        self.target = target


class PairwiseRecords(BaseRecords):
    """Pairwise records of queries associated with (positive, negative) pairs"""

    # The queries
    _topics: List[IDTextRecord]

    # The document IDs (positive)
    positives: List[IDTextRecord]

    # The scores of the retriever
    negatives: List[IDTextRecord]

    def __init__(self):
        self._topics = []
        self.positives = []
        self.negatives = []

    def add(self, record: PairwiseRecord):
        self._topics.append(record.query)
        self.positives.append(record.positive)
        self.negatives.append(record.negative)

    @property
    def topics(self):
        return itertools.chain(self._topics, self._topics)

    def set_unique_topics(self, topics: List[IDTextRecord]):
        assert len(topics) == len(self._topics), (
            f"Number of topics do not match ({len(topics)} vs {len(self._topics)})"
        )
        self._topics = topics

    def set_unique_documents(self, documents: List[IDTextRecord]):
        N = len(self._topics)
        assert len(documents) == N * 2
        self.positives = documents[:N]
        self.negatives = documents[N:]

    queries = topics

    @property
    def unique_topics(self):
        return self._topics

    unique_queries = unique_topics

    @property
    def documents(self):
        return itertools.chain(self.positives, self.negatives)

    def pairs(self):
        """Returns the list of query/document indices for which we should compute the score, or None if all (cartesian product). This method should be
        used with `unique_topics`"""
        indices = list(range(len(self._topics)))
        return indices * 2, list(range(2 * len(self.positives)))

    def __len__(self):
        return len(self._topics)

    def __getitem__(self, ix: Union[slice, int]):
        if isinstance(ix, slice):
            records = PairwiseRecords()
            for i in range(ix.start, min(ix.stop, len(self._topics)), ix.step or 1):
                records.add(
                    PairwiseRecord(
                        self._topics[i], self.positives[i], self.negatives[i]
                    )
                )
            return records

        return PairwiseRecord(self._topics[ix], self.positives[ix], self.negatives[ix])


class PairwiseRecordsWithTarget(PairwiseRecords):
    """Pairwise records associated with a label (saying which document is better)"""

    target: List[int]

    def __init__(self):
        super().__init__()
        self.target = []

    def add(self, record: PairwiseRecordWithTarget):
        self._topics.append(record.query)
        self.positives.append(record.positive)
        self.negatives.append(record.negative)
        self.target.append(record.target)

    def get_target(self):
        return self.target

    def __getitem__(self, ix: Union[slice, int]):
        if isinstance(ix, slice):
            records = PairwiseRecordsWithTarget()
            for i in range(ix.start, min(ix.stop, len(self._topics)), ix.step or 1):
                records.add(
                    PairwiseRecordWithTarget(
                        self._topics[i],
                        self.positives[i],
                        self.negatives[i],
                        self.target[i],
                    )
                )
            return records

        return PairwiseRecordWithTarget(
            self._topics[ix], self.positives[ix], self.negatives[ix], self.target[ix]
        )


class ListwiseRecord(Generic[DocT, QueryT]):
    """A listwise record is composed of a query and a list of documents"""

    query: QueryT
    documents: List[DocT]

    def __init__(self, query: QueryT, documents: List[DocT]):
        self.query = query
        self.documents = documents

    def get_queries(self) -> List[QueryT]:
        return [self.query]

    def with_queries(self, qs: "List[QueryT2]") -> "ListwiseRecord[DocT, QueryT2]":
        return ListwiseRecord(qs[0], self.documents)

    def get_documents(self) -> List[DocT]:
        return self.documents

    def with_documents(self, ds: "List[DocT2]") -> "ListwiseRecord[DocT2, QueryT]":
        return ListwiseRecord(self.query, list(ds))


class ListwiseRecords(BaseRecords):
    """Listwise records of queries associated with lists of documents"""

    # The queries
    _topics: List[IDTextRecord]

    # The list of documents per query
    _documents: List[List[IDTextRecord]]

    def __init__(self):
        self._topics = []
        self._documents = []

    def add(self, record: ListwiseRecord):
        self._topics.append(record.query)
        self._documents.append(record.documents)

    @property
    def topics(self):
        return itertools.chain(self._topics, self._topics)

    def set_unique_topics(self, topics: List[IDTextRecord]):
        assert len(topics) == len(self._topics), (
            f"Number of topics do not match ({len(topics)} vs {len(self._topics)})"
        )
        self._topics = topics

    def set_unique_documents(self, documents: List[IDTextRecord]):
        raise NotImplementedError("set_unique_documents() in ListwiseRecords")

    queries = topics

    @property
    def unique_topics(self):
        return self._topics

    unique_queries = unique_topics

    @property
    def documents(self):
        return itertools.chain.from_iterable(self._documents)

    def pairs(self):
        indices = list(range(len(self._topics)))
        return indices * 2, list(range(2 * len(self._documents)))

    def __len__(self):
        return len(self._topics)

    def __getitem__(self, ix: Union[slice, int]):
        if isinstance(ix, slice):
            records = ListwiseRecords()
            for i in range(ix.start, min(ix.stop, len(self._topics)), ix.step or 1):
                records.add(ListwiseRecord(self._topics[i], self._documents[i]))
            return records

        return ListwiseRecord(self._topics[ix], self._documents[ix])


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

        _topics: The list of queries
        _documents: The list of documents
        _relevances: (query x document) matrix with relevance score (between 0 and 1)
    """

    _topics: List[IDTextRecord]
    """The list of queries to score"""

    _documents: List[IDTextRecord]
    """The list of documents to score"""

    relevances: torch.Tensor
    """A 2D tensor (query x document) indicating the relevance of the each
    query/document pair"""

    is_product = True

    def __init__(self):
        self._topics = []
        self._documents = []

    def add_topics(self, *topics: IDTextRecord):
        self._topics.extend(topics)

    def add_documents(self, *documents: IDTextRecord):
        self._documents.extend(documents)

    def set_relevances(self, relevances: torch.Tensor):
        assert relevances.shape[0] == len(self._topics), (
            f"The number of queries {len(self._topics)} "
            + "does not match the number of rows {relevances.shape[0]}"
        )
        assert relevances.shape[1] == len(self._documents), (
            f"The number of documents {len(self._documents)} "
            + "does not match the number of columns {relevances.shape[1]}"
        )
        self.relevances = relevances

    def __len__(self):
        return len(self._topics)

    @property
    def topics(self):
        for q in self._topics:
            for _ in self._documents:
                yield q

    queries = topics

    @property
    def unique_topics(self):
        return self._topics

    unique_queries = unique_topics

    @property
    def documents(self):
        for _ in self._topics:
            for d in self._documents:
                yield d

    @property
    def unique_documents(self) -> Iterable[IDTextRecord]:
        return self._documents

    def pairs(self) -> Tuple[Iterable[int], Iterable[int]]:
        topics = []
        documents = []
        for q in range(len(self._topics)):
            for d in range(len(self._documents)):
                topics.append(q)
                documents.append(d)
        return topics, documents

    def __getitem__(self, ix: Union[slice, int]):
        if isinstance(ix, slice):
            start, stop, step = ix.indices(len(self._topics))
            records = ProductRecords()
            # add selected topics
            for i in range(start, stop, step):
                records.add_topics(self._topics[i])
            # keep same documents
            for d in self._documents:
                records.add_documents(d)
            # slice relevances rows if present
            if hasattr(self, "relevances") and self.relevances is not None:
                # build list of row indices and index into tensor
                rows = list(range(start, stop, step))
                records.set_relevances(self.relevances[rows])
            return records

        # integer index: return a PointwiseRecords containing this topic
        # paired with every document (preserving relevances when present)
        if ix < 0:
            ix += len(self._topics)
        if ix < 0 or ix >= len(self._topics):
            raise IndexError("ProductRecords index out of range")

        records = PointwiseRecords()
        topic = self._topics[ix]

        if hasattr(self, "relevances") and self.relevances is not None:
            row = self.relevances[ix]
            try:
                rels = list(row.tolist())
            except Exception:
                rels = list(row)

            for d, r in zip(self._documents, rels):
                records.add(PointwiseRecord(topic, d, float(r)))
        else:
            for d in self._documents:
                records.add(PointwiseRecord(topic, d, None))

        return records


class DocumentRecords(List[IDTextRecord]):
    """Masked Language Modeling Records are a set of documents"""

    # Text of the documents
    documents: List[IDTextRecord]

    def __init__(self):
        super().__init__()
        self.documents = []

    def add(self, record: IDTextRecord):
        self.documents.append(record)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, ix: Union[slice, int]):
        if isinstance(ix, slice):
            records = DocumentRecords()
            for i in range(ix.start, min(ix.stop, len(self)), ix.step or 1):
                records.add(self.documents[i])
            return records

        return DocumentRecords(self.documents[ix])

    @staticmethod
    def from_texts(
        documents: List[str],
    ):
        records = DocumentRecords()
        records.documents = list(documents)
        return records

    def to_texts(self) -> List[str]:
        texts = []
        for doc in self.documents:
            texts.append(doc.document["text_item"].text)

        return texts
