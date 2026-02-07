import torch
import itertools
from datamaestro_text.data.ir import (
    TopicRecord,
    DocumentRecord,
    TextItem,
    create_record,
)
from typing import (
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)


class PointwiseRecord:
    """A record from a pointwise sampler"""

    # The query
    topic: TopicRecord

    # The document
    document: DocumentRecord

    # The relevance
    relevance: Optional[float]

    def __init__(
        self,
        topic: TopicRecord,
        document: DocumentRecord,
        relevance: Optional[float] = None,
    ):
        self.topic = topic
        self.document = document
        self.relevance = relevance

    @property
    def query(self):
        return self.topic


RT = TypeVar("RT")


class BaseRecords(List[RT]):
    """Base records just exposes iterables on (query, document) pairs

    Records can be structured, i.e. the same queries and documents
    can be used more than once. To allow optimization (e.g. pre-computing
    document/query representation),
    """

    topics: Iterable[TopicRecord]
    documents: Iterable[DocumentRecord]
    is_product = False

    @property
    def unique_topics(self) -> Iterable[TopicRecord]:
        return self.topics

    unique_queries = unique_topics

    @property
    def unique_documents(self) -> Iterable[DocumentRecord]:
        return self.documents

    @property
    def queries(self):
        """Deprecated: use topics"""
        return self.topics

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
    topics: List[TopicRecord]

    # Text of the documents
    documents: List[DocumentRecord]

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
        records.topics = list(map(lambda t: create_record(text=t), topics))
        records.documents = list(map(lambda t: create_record(text=t), documents))
        records.relevances = relevances
        return records


class PairwiseRecord:
    """A pairwise record is composed of a query, a positive and a negative document"""

    query: TopicRecord
    positive: DocumentRecord
    negative: DocumentRecord

    def __init__(
        self, query: TopicRecord, positive: DocumentRecord, negative: DocumentRecord
    ):
        self.query = query
        self.positive = positive
        self.negative = negative


class PairwiseRecordWithTarget(PairwiseRecord):
    """A pairwise record is composed of a query, a positive and a negative
    document, and the indetifier which says the one on the first is pos or
    neg"""

    target: int

    def __init__(
        self,
        query: TopicRecord,
        positive: DocumentRecord,
        negative: DocumentRecord,
        target: int,
    ):
        super().__init__(query, positive, negative)
        self.target = target


class PairwiseRecords(BaseRecords):
    """Pairwise records of queries associated with (positive, negative) pairs"""

    # The queries
    _topics: List[TopicRecord]

    # The document IDs (positive)
    positives: List[DocumentRecord]

    # The scores of the retriever
    negatives: List[DocumentRecord]

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

    def set_unique_topics(self, topics: List[TopicRecord]):
        assert len(topics) == len(
            self._topics
        ), f"Number of topics do not match ({len(topics)} vs {len(self._topics)})"
        self._topics = topics

    def set_unique_documents(self, documents: List[DocumentRecord]):
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
    
class ListwiseRecord:
    """A listwise record is composed of a query and a list of documents"""

    query: TopicRecord
    documents: List[DocumentRecord]

    def __init__(
        self, query: TopicRecord, documents: List[DocumentRecord]
    ):
        self.query = query
        self.documents = documents

class ListwiseRecords(BaseRecords):
    """Listwise records of queries associated with lists of documents"""

    # The queries
    _topics: List[TopicRecord]

    # The list of documents per query
    _documents: List[List[DocumentRecord]]


    def __init__(self):
        self._topics = []
        self._documents = []

    def add(self, record: ListwiseRecord):
        self._topics.append(record.query)
        self._documents.append(record.documents)

    @property
    def topics(self):
        return itertools.chain(self._topics, self._topics)

    def set_unique_topics(self, topics: List[TopicRecord]):
        assert len(topics) == len(
            self._topics
        ), f"Number of topics do not match ({len(topics)} vs {len(self._topics)})"
        self._topics = topics

    def set_unique_documents(self, documents: List[DocumentRecord]):
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
                records.add(
                    ListwiseRecord(
                        self._topics[i], self._documents[i]
                    )
                )
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

    _topics: List[TopicRecord]
    """The list of queries to score"""

    _documents: List[DocumentRecord]
    """The list of documents to score"""

    relevances: torch.Tensor
    """A 2D tensor (query x document) indicating the relevance of the each
    query/document pair"""

    is_product = True

    def __init__(self):
        self._topics = []
        self._documents = []

    def add_topics(self, *topics: TopicRecord):
        self._topics.extend(topics)

    def add_documents(self, *documents: DocumentRecord):
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
    def unique_documents(self):
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


class DocumentRecords(List[DocumentRecord]):
    """Masked Language Modeling Records are a set of documents"""

    # Text of the documents
    documents: List[DocumentRecord]

    def __init__(self):
        super().__init__()
        self.documents = []

    def add(self, record: DocumentRecord):
        self.documents.append(record)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, ix: Union[slice, int]):
        if isinstance(ix, slice):
            records = DocumentRecord()
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
            texts.append(doc.document[TextItem].text)

        return texts
