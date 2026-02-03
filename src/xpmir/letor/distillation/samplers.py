from typing import (
    Iterable,
    Iterator,
    NamedTuple,
    Tuple,
    List,
    Any,
)
import numpy as np

from experimaestro import Config, Meta, Param
from datamaestro.data import File
from datamaestro_text.data.ir.base import (
    TopicRecord,
    DocumentRecord,
    ScoredItem,
    SimpleTextItem,
    IDItem,
    create_record,
)
from xpm_torch.utils.iter import (
    SerializableIterator,
    SkippingIterator,
    SerializableIteratorTransform,
)

from xpm_torch.base import Sampler
from xpmir.letor.samplers.hydrators import SampleHydrator
from xpmir.rankers import ScoredDocument


class PairwiseDistillationSample(NamedTuple):
    query: TopicRecord
    """The query"""

    documents: Tuple[DocumentRecord, DocumentRecord]
    """Positive/negative document with teacher scores"""


class PairwiseDistillationSamples(Config, Iterable[PairwiseDistillationSample]):
    """Pairwise distillation file"""

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        raise NotImplementedError()


class PairwiseHydrator(PairwiseDistillationSamples, SampleHydrator):
    """Hydrate ID-based samples with document and/or query content"""

    samples: Param[PairwiseDistillationSamples]
    """The distillation samples without texts for query and documents"""

    def transform(self, sample: PairwiseDistillationSample):
        topic, documents = sample.query, sample.documents

        if transformed := self.transform_topics([topic]):
            topic = transformed[0]

        if transformed := self.transform_documents(documents):
            documents = tuple(
                ScoredDocument(d, sd[ScoredItem].score)
                for d, sd in zip(transformed, sample.documents)
            )

        return PairwiseDistillationSample(topic, documents)

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        iterator = iter(self.samples)
        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform
        )


class PairwiseDistillationSamplesTSV(PairwiseDistillationSamples, File):
    """A TSV file (Score 1, Score 2, Query, Document 1, Document 2)"""

    with_docid: Meta[bool]
    with_queryid: Meta[bool]

    def __iter__(self) -> Iterator[PairwiseDistillationSample]:
        return self.iter()

    def iter(self) -> Iterator[PairwiseDistillationSample]:
        import csv

        def iterate():
            with self.path.open("rt") as fp:
                for row in csv.reader(fp, delimiter="\t"):
                    if self.with_queryid:
                        query = create_record(id=row[2])
                    else:
                        query = create_record(text=row[2])

                    if self.with_docid:
                        documents = (
                            DocumentRecord(IDItem(row[3]), ScoredItem(float(row[0]))),
                            DocumentRecord(IDItem(row[4]), ScoredItem(float(row[1]))),
                        )
                    else:
                        documents = (
                            DocumentRecord(
                                SimpleTextItem(row[3]), ScoredItem(float(row[0]))
                            ),
                            DocumentRecord(
                                SimpleTextItem(row[4]), ScoredItem(float(row[1]))
                            ),
                        )

                    yield PairwiseDistillationSample(query, documents)

        return SkippingIterator(iterate())


class _DistillationPairwiseBatchIterator(SerializableIterator):
    """Batch iterator for DistillationPairwiseSampler.

    Defined at module level to allow pickling for multiprocessing.
    """

    def __init__(self, sampler: "DistillationPairwiseSampler", size: int):
        self.sampler = sampler
        self.size = size
        self._iter = None
        self._state = None

    def _ensure_iter(self):
        """Lazily initialize the iterator."""
        if self._iter is None:
            self._iter = self.sampler.pairwise_iter()
            if self._state is not None:
                self._iter.load_state_dict(self._state)
                self._state = None

    def __getstate__(self):
        """For pickling: save the state_dict instead of the iterator."""
        state = self.__dict__.copy()
        if self._iter is not None:
            state["_state"] = self._iter.state_dict()
        state["_iter"] = None
        return state

    def __setstate__(self, state):
        """For unpickling: restore from state_dict."""
        self.__dict__.update(state)

    def state_dict(self):
        self._ensure_iter()
        return self._iter.state_dict()

    def load_state_dict(self, state):
        if self._iter is not None:
            self._iter.load_state_dict(state)
        else:
            self._state = state

    def __next__(self):
        self._ensure_iter()
        batch = []
        for _, record in zip(range(self.size), self._iter):
            batch.append(record)
        return batch


class DistillationPairwiseSampler(Sampler):
    """Just loops over samples"""

    samples: Param[PairwiseDistillationSamples]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

    def pairwise_iter(self) -> SerializableIterator[PairwiseDistillationSample, Any]:
        return SkippingIterator.make_serializable(iter(self.samples))

    def pairwise_batch_iter(
        self, size
    ) -> SerializableIterator[List[PairwiseDistillationSample], Any]:
        """Batchwise iterator

        Can be subclassed by some classes to be more efficient"""
        return _DistillationPairwiseBatchIterator(self, size)


#######
# LISTWISE Distillation datasets samplers
######

class ListwiseDistillationSample(NamedTuple):
    query: TopicRecord
    """The query"""

    documents: List[DocumentRecord]
    """List of documents with their ranking position"""


class ListwiseDistillationSamples(Config, Iterable[ListwiseDistillationSample]):
    """Listwise distillation file"""

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        raise NotImplementedError()


class ListwiseHydrator(ListwiseDistillationSamples, SampleHydrator):
    """Hydrate ID-based samples with document and/or query content"""

    samples: Param[ListwiseDistillationSamples]
    """The distillation samples without texts for query and documents"""

    def transform(self, sample: ListwiseDistillationSample):
        topic, documents = sample.query, sample.documents

        if transformed := self.transform_topics([topic]):
            topic = transformed[0]

        if transformed := self.transform_documents(documents):
            documents = list(
                ScoredDocument(d, sd[ScoredItem].score)
                for d, sd in zip(transformed, sample.documents)
            )

        return ListwiseDistillationSample(topic, documents)

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        iterator = iter(self.samples)
        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform
        )

class ListwiseDistillationSamplesTSV(ListwiseDistillationSamples, File):
    """A TSV file ("query_id", "q0", "doc_id", "rank", "score", "system")"""

    top_k: Meta[int]
    with_docid: Meta[bool]
    with_queryid: Meta[bool]

    def __iter__(self) -> Iterator[ListwiseDistillationSample]:
        return self.iter()

    def iter(self) -> Iterator[ListwiseDistillationSample]:
        import csv

        def iterate():
            with self.path.open("rt") as fp:
                reader = csv.reader(fp, delimiter="\t")

                current_q = None
                current_query_record = None
                documents: List[DocumentRecord] = []

                for row in reader:
                    if not row:
                        continue

                    qkey = row[0]

                    # start a new block when query changes or when we've reached top_k
                    if current_q is None:
                        current_q = qkey
                        current_query_record = (
                            create_record(id=qkey)
                            if self.with_queryid
                            else create_record(text=qkey)
                        )

                    if qkey != current_q or (self.top_k and len(documents) >= int(self.top_k)):
                        if documents:
                            yield ListwiseDistillationSample(current_query_record, documents)
                        # reset for new block (may be new query or another block for same query)
                        current_q = qkey
                        current_query_record = (
                            create_record(id=qkey)
                            if self.with_queryid
                            else create_record(text=qkey)
                        )
                        documents = []

                    if self.with_docid:
                        doc = DocumentRecord(IDItem(row[2]), ScoredItem(float(row[4])))
                    else:
                        doc = DocumentRecord(SimpleTextItem(row[2]), ScoredItem(float(row[4])))

                    documents.append(doc)

                # emit any remaining documents for the last block
                if documents:
                    yield ListwiseDistillationSample(current_query_record, documents)

        return SkippingIterator(iterate())


class _DistillationListwiseBatchIterator(SerializableIterator):
    """Batch iterator for DistillationListwiseSampler.

    Defined at module level to allow pickling for multiprocessing.
    """

    def __init__(self, sampler: "DistillationListwiseSampler", size: int):
        self.sampler = sampler
        self.size = size
        self._iter = None
        self._state = None

    def _ensure_iter(self):
        """Lazily initialize the iterator."""
        if self._iter is None:
            self._iter = self.sampler.listwise_iter()
            if self._state is not None:
                self._iter.load_state_dict(self._state)
                self._state = None

    def __getstate__(self):
        """For pickling: save the state_dict instead of the iterator."""
        state = self.__dict__.copy()
        if self._iter is not None:
            state["_state"] = self._iter.state_dict()
        state["_iter"] = None
        return state

    def __setstate__(self, state):
        """For unpickling: restore from state_dict."""
        self.__dict__.update(state)

    def state_dict(self):
        self._ensure_iter()
        return self._iter.state_dict()

    def load_state_dict(self, state):
        if self._iter is not None:
            self._iter.load_state_dict(state)
        else:
            self._state = state

    def __next__(self):
        self._ensure_iter()
        batch = []
        for _, record in zip(range(self.size), self._iter):
            batch.append(record)
        return batch


class DistillationListwiseSampler(Sampler):
    """Just loops over samples"""

    samples: Param[ListwiseDistillationSamples]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

    def listwise_iter(self) -> SerializableIterator[ListwiseDistillationSample, Any]:
        return SkippingIterator.make_serializable(iter(self.samples))

    def listwise_batch_iter(
        self, size
    ) -> SerializableIterator[List[ListwiseDistillationSample], Any]:
        """Batchwise iterator

        Can be subclassed by some classes to be more efficient"""
        return _DistillationListwiseBatchIterator(self, size)