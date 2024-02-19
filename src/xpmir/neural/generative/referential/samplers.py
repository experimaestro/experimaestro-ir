from typing import Any, Optional, TypeVar, List, Iterator, Union
from pathlib import Path
from attrs import define
import json
from experimaestro.compat import cached_property
import random

import numpy as np
from experimaestro import Param, Config
from datamaestro_text.data.ir import DocumentStore
from datamaestro_text.data.ir.base import Document, IDDocument
from xpmir.learning import Sampler
from xpmir.letor.records import PairwiseRecord, DocumentRecord
from xpmir.letor.samplers import PairwiseSampler
from xpmir.letor.samplers.hydrators import SampleTransform
from xpmir.utils.iter import (
    SerializableIterator,
    RandomStateSerializableAdaptor,
    SkippingIterator,
    InfiniteSkippingIterator,
    iterable_of,
    SerializableIteratorTransform,
)

State = TypeVar("State")
T = TypeVar("T")
RT = TypeVar("RT")


# deprecated
class AbstractDynamicNegativesSampler(PairwiseSampler):
    """Specific for referential"""

    documents: Param[DocumentStore]
    """The document store where the sampler is based on"""

    base_sampler: Param[PairwiseSampler]
    """The base sampler from which we try to extract the negatives"""

    max_depth: Param[int]
    """The number of the maximum depth for the model"""

    dimension: Param[int]
    """The number of dimension per layer, not including the eos"""

    max_docs_per_seq_coeff: Param[float] = 1.5
    """The number of dimension per layer is the average possible documents *
    this coefficient"""

    fake_average_coeff: Param[float] = 0.8
    """During the update of the log_proba, we consider the new threshold as
    fake_average_coeff * previous + (1 - fake_average_coeff) * new_input"""


# deprecated
class DynamicNegativesSampler(AbstractDynamicNegativesSampler):
    """The random span sampler where the negatives are sampled in an
    hierarchical way, designed for REFERENTIAL pretraining.

    During the pretraining there is not early eos, so the number of the
    possible sequence is dimension^max_depth, and store them inside a matrix
    of dimension max_depth.

    Only support the pairwise operation for the moment

    Similar to the previous sampler, but only support for the full depth negative mining
    """

    def initialize(self, random: Optional[np.random.RandomState] = None):
        super().initialize(random)
        self.base_sampler.initialize(random)
        # FIXME： Better to make the instance created inside the pairwise_iter
        self.iterator = ReferentialRandomStateSerializableAdapter(
            self.base_sampler.pairwise_iter(),
            self.random,
            self.documents,
            self.max_depth,
            self.dimension,
            self.max_docs_per_seq_coeff,
            self.fake_average_coeff,
        )

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord, Any]:
        return self.iterator


# deprecated
class ReferentialRandomStateSerializableAdapter(
    RandomStateSerializableAdaptor[SerializableIterator[PairwiseRecord]]
):
    id_matrix: np.chararray
    """A matrix of shape [dim, dim, ..., dim, N_2] of dim max_depth + 1, which
    contains the id of the documents"""

    log_proba_mean_matrix: np.array
    """A matrix of shape [dim, dim, ..., dim, N_2] of dim max_depth + 1, which
    contains the id of the documents
    It contains the avg log_probability of the given sequence"""

    def __init__(
        self,
        iterator: SerializableIterator[PairwiseRecord],
        random: np.random.RandomState,
        documents: DocumentStore,
        max_depth: int,
        dimension: int,
        max_docs_per_seq_coeff: float,
        fake_average_coeff: float,
    ):
        """Initialize the id_matrix and id_proba matrix"""
        super().__init__(iterator)
        super().set_random(random)
        self.documents = documents
        self.max_depth = max_depth
        self.fake_average_coeff = fake_average_coeff
        sequence_num = dimension**self.max_depth
        nb_docs = int(
            self.documents.documentcount / sequence_num * max_docs_per_seq_coeff
        )

        shape_proba = [dimension for _ in range(self.max_depth)]
        shape_ids = shape_proba + [nb_docs]
        # initialize the ids with -1.
        self.id_matrix = np.empty(shape_ids, dtype="<U10")
        # initialize the log_probabilities with the a small probability
        self.log_proba_mean_matrix = np.full(shape_proba, -10.0)

    def update_matrix(
        self,
        sampled_tokens: np.array,  # shape [max_depth, bs]
        ids: np.chararray,  # shape: bs
        log_proba: np.array,  # shape: bs
    ):
        """Update the id_matrix and log_proba_matrix of the negatives"""
        # the previous average probability for the current given ids
        log_proba_mean = self.log_proba_mean_matrix[tuple(sampled_tokens)]
        # update the means(using a fake average)
        self.log_proba_mean_matrix[tuple(sampled_tokens)] = (
            log_proba_mean * self.fake_average_coeff
            + (1 - self.fake_average_coeff) * log_proba
        )

        # get the ids and the sequences which is better than the average
        better_indices = np.where(log_proba > log_proba_mean)[0]
        better_ids = ids[better_indices]
        better_sampled_tokens = sampled_tokens[:, better_indices]

        # shape [nb_better, nb_documents]
        ids_sequences = self.id_matrix[tuple(better_sampled_tokens)]

        # randomly replace the previous ids
        indices_doc_to_replace = np.random.randint(
            self.id_matrix.shape[-1],
            size=(ids_sequences.shape[0]),
        )

        # replace them!
        self.id_matrix[
            tuple(np.vstack((better_sampled_tokens, indices_doc_to_replace)))
        ] = better_ids

    def load_state_dict(self, state: Any):
        # state = iterator state + negative states.
        self.id_matrix = state["negative_state"]["id_matrix"]
        self.log_proba_mean_matrix = state["negative_state"]["id_proba"]
        self.iterator.load_state_dict(state["iterator_state"])

    def state_dict(self) -> Any:
        iterator_state = self.iterator.state_dict()
        negative_state = {
            "id_matrix": self.id_matrix,
            "id_proba": self.log_proba_mean_matrix,
        }
        return {
            "iterator_state": iterator_state,
            "negative_state": negative_state,
        }

    def hierarchical_neg_mining(
        self,
        target_id: str,  # the string id
    ):
        """return the internal id of the negative, if not find, return None"""
        indices = np.where(self.id_matrix == target_id)
        if indices[0].shape[0] == 0:  # target id not found in matrix
            return None

        # random choose one to as the target sequence
        random_choosed = np.vstack(indices)[:, self.random.randint(indices[0].shape[0])]
        # build the hierarchical matrix to search
        # e.g. the input is at [7,8,2,6], first we search from [7,8,2],
        # if not found, we go to [7, 8], etc
        for i in range(self.max_depth, 0, -1):
            hierarchical_targets = self.id_matrix[tuple(random_choosed[:i])]
            satisfied = np.where(
                np.logical_and(
                    hierarchical_targets != "",
                    hierarchical_targets != target_id,
                )
            )
            if hierarchical_targets[satisfied].shape[0] > 0:  # found at this level
                return self.random.choice(hierarchical_targets[satisfied])

        return None  # not found over all the hierarchical matrix

    def __next__(self) -> SerializableIterator[PairwiseRecord, Any]:
        original_pair: PairwiseRecord = next(self.iterator)
        id_pos_qry = original_pair.positive.document.get_id()
        res = self.hierarchical_neg_mining(id_pos_qry)
        if res:
            record_neg = DocumentRecord(self.documents.document_ext(res))
        else:
            record_neg = original_pair.negative

        return PairwiseRecord(original_pair.query, original_pair.positive, record_neg)


# deprecated
# FIXME: Not working for the moment, need to treat the current depth
class DepthUpdatableDynamicNegativesSampler(AbstractDynamicNegativesSampler):
    """The random span sampler where the negatives are sampled in an
    hierarchical way, designed for REFERENTIAL pretraining.

    During the pretraining there is not early eos, so the number of the
    possible sequence is dimension^max_depth, and store them inside a matrix
    of dimension max_depth.

    Only support the pairwise operation for the moment

    The training is based on the depth updable.
    """

    def initialize(self, random: Optional[np.random.RandomState] = None):
        super().initialize(random)
        self.base_sampler.initialize(random)
        self.iterator = DepthUpdatableReferentialRandomStateSerializableAdapter(
            self.base_sampler.pairwise_iter(),
            self.random,
            self.documents,
            self.max_depth,
            self.dimension,
            self.max_docs_per_seq_coeff,
            self.fake_average_coeff,
        )

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord, Any]:
        return self.iterator


# deprecated
class DepthUpdatableReferentialRandomStateSerializableAdapter(
    RandomStateSerializableAdaptor[SerializableIterator[PairwiseRecord]]
):
    id_matrix: list[np.chararray]
    """A list of length max_depth - 1(for the last layer this matrix is
    useless), where each element is of dimension depth+1, like
    ([dim, N_1], [dim, dim, N_2], etc) which contains the id of the documents"""

    log_proba_mean_matrix: list[np.array]
    """A list of the same length of id_matrix, and each element of shape
    [dim], [dim, dim], etc... .
    It contains the log_probability of the given sequence"""

    current_depth: int
    """Indicate the current depth of the model"""

    def __init__(
        self,
        iterator: SerializableIterator[SerializableIterator[PairwiseRecord, Any], Any],
        random: np.random.RandomState,
        documents: DocumentStore,
        max_depth: int,
        dimension: int,
        max_docs_per_seq_coeff: float,
        fake_average_coeff: float,
    ):
        super().__init__(iterator)
        super().set_random(random)
        self.documents = documents
        self.max_depth = max_depth
        self.fake_average_coeff = fake_average_coeff
        self.id_matrix = []
        self.log_proba_mean_matrix = []

        for depth in range(1, self.max_depth):
            sequence_num = dimension**depth
            nb_docs = int(
                self.documents.documentcount / sequence_num * max_docs_per_seq_coeff
            )
            shape_mean = [dimension for _ in range(depth)]
            shape_ids = shape_mean + [nb_docs]
            # initialize the ids with "".
            self.id_matrix.append(np.empty(shape_ids, dtype="<U10"))
            # initialize the log_probabilities with the a small probability
            self.log_proba_mean_matrix.append(np.full(shape_mean, -10.0))

    def update_matrix(
        self,
        sampled_tokens: np.array,  # shape [depth, bs]
        ids: np.chararray,  # shape: bs
        log_proba: np.array,  # shape: bs
    ):
        current_depth, _ = sampled_tokens.shape

        # the previous average probability for the current given ids
        log_proba_mean = self.log_proba_mean_matrix[current_depth - 1][
            tuple(sampled_tokens)
        ]
        # update the means(using a fake average)
        self.log_proba_mean_matrix[current_depth - 1][tuple(sampled_tokens)] = (
            log_proba_mean * self.fake_average_coeff
            + (1 - self.fake_average_coeff) * log_proba
        )

        # get the ids and the sequences which is better than the average
        better_indices = np.where(log_proba > log_proba_mean)[0]
        better_ids = ids[better_indices]
        better_sampled_tokens = sampled_tokens[:, better_indices]

        # shape [nb_better, nb_documents]
        ids_sequences = self.id_matrix[current_depth - 1][tuple(better_sampled_tokens)]

        # randomly replace the previous ids
        indices_doc_to_replace = np.random.randint(
            self.id_matrix[current_depth - 1].shape[-1],
            size=(ids_sequences.shape[0],),
        )

        # replace them!
        self.id_matrix[current_depth - 1][
            tuple(np.vstack((better_sampled_tokens, indices_doc_to_replace)))
        ] = better_ids

    def hierarchical_neg_mining(
        self,
        id_matrix: np.chararray,  # shape depends on the layer
        target_id: str,
    ):
        """return the internal id of the negative, if not find, return None"""
        indices = np.where(id_matrix == target_id)
        if indices[0].shape[0] == 0:  # target id not found in matrix
            return None

        # random choose one to as the target sequence
        random_choosed = np.vstack(indices)[:, self.random.randint(indices[0].shape[0])]
        # build the hierarchical matrix to search
        # e.g. the input is at [7,8,2,6], first we search from [7,8,2],
        # if not found, we go to [7, 8], etc
        for i in range(self.current_depth, 0, -1):
            hierarchical_targets = id_matrix[tuple(random_choosed[:i])]
            satisfied = np.where(
                np.logical_and(
                    hierarchical_targets != "", hierarchical_targets != target_id
                )
            )
            if hierarchical_targets[satisfied].shape[0] > 0:  # found at this level
                return self.random.choice(hierarchical_targets[satisfied])

        return None  # not found over all the hierarchical matrix

    def load_state_dict(self, state: Any):
        # state = iterator state + negative states.
        self.id_matrix = state["negative_state"]["id_matrix"]
        self.log_proba_mean_matrix = state["negative_state"]["id_proba"]
        self.iterator.load_state_dict(state["iterator_state"])

    def state_dict(self) -> Any:
        iterator_state = self.iterator.state_dict()
        negative_state = {
            "id_matrix": self.id_matrix,
            "id_proba": self.log_proba_mean_matrix,
        }
        return {
            "iterator_state": iterator_state,
            "negative_state": negative_state,
        }

    def __next__(self) -> SerializableIterator[PairwiseRecord, Any]:
        original_pair: PairwiseRecord = next(self.iterator)
        id_pos_qry = original_pair.positive.document.get_id()
        if self.current_depth > 1:
            # current depth = 2 so use the prefix at depth 1, so indice = 0
            id_matrix = self.id_matrix[self.current_depth - 2]
            res = self.hierarchical_neg_mining(id_matrix, id_pos_qry)
            if res:
                record_neg = DocumentRecord(self.documents.document_ext(res))
            else:
                record_neg = original_pair.negative
        else:
            record_neg = original_pair.negative
        return PairwiseRecord(original_pair.query, original_pair.positive, record_neg)


# TODO: Put it to the record, and can make it more generic: list[int] ->
# list[str] so we can tokenize them
class ReferentialDocumentIDRecord:

    document: DocumentRecord

    target: List[int]

    def __init__(self, document: DocumentRecord, target: List[int]):
        self.document = document
        self.target = target


class ReferentialDocumentIDRecords(List[RT]):

    documents: List[DocumentRecord]

    targets: List[List[int]]

    def __init__(self):
        self.documents = []
        self.targets = []

    def add(self, record: ReferentialDocumentIDRecord):
        self.documents.append(record.document)
        self.targets.append(record.target)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, ix: Union[slice, int]):
        if isinstance(ix, slice):
            records = ReferentialDocumentIDRecords()
            for i in range(ix.start, min(ix.stop, len(self.documents)), ix.step or 1):
                records.add(
                    ReferentialDocumentIDRecord(self.documents[i], self.targets[i])
                )
            return records

        return ReferentialDocumentIDRecord(self.documents[ix], self.targets[ix])


@define(kw_only=True)
class JSONLReferentialDocumentIdSample:

    documents: List[Document]
    """The list of document to given id"""

    ids: List[int]
    """The sequences"""


class JSONLReferentialDocumentIdDataset(Config):

    path: Param[Path]
    """The path to the Jsonl file"""

    def iter(self) -> Iterator[JSONLReferentialDocumentIdSample]:
        with self.path.open("r") as fp:
            for line in fp:
                sample = json.loads(line)
                documents = [
                    IDDocument(id=document_id)
                    for document_id in list(sample.values())[0]
                ]
                id_list = list(sample.keys())[0].split("\t")
                if id_list == [""]:
                    # if the id_list = [""] (correspond to eos directly)
                    id_list = []
                else:
                    id_list = [int(id) for id in id_list]
                yield JSONLReferentialDocumentIdSample(documents=documents, ids=id_list)


class ReferentialSampler(Sampler):
    def referential_iter(self) -> SkippingIterator[ReferentialDocumentIDRecord]:
        pass


class JSONLReferentialDocumentIdSampler(ReferentialSampler):

    dataset: Param[JSONLReferentialDocumentIdDataset]
    """The dataset store"""

    @cached_property
    def random_data_list(self) -> List[JSONLReferentialDocumentIdSample]:
        dataset_list = list(self.dataset.iter())
        random.shuffle(dataset_list)
        return dataset_list

    def referential_iter(self) -> SkippingIterator[ReferentialDocumentIDRecord]:
        class _Iterator(
            RandomStateSerializableAdaptor[
                SerializableIterator[JSONLReferentialDocumentIdSample]
            ]
        ):
            def __init__(
                self,
                iterator: SerializableIterator[JSONLReferentialDocumentIdSample],
                random: np.random.RandomState,
            ):
                super().__init__(iterator)
                self.random = random

            def __next__(self):
                sample = next(self.iterator)
                doc = sample.documents[self.random.randint(len(sample.documents))]
                return ReferentialDocumentIDRecord(
                    document=DocumentRecord(doc), target=sample.ids
                )

        base = InfiniteSkippingIterator(
            iterable_of(lambda: iter(self.random_data_list))
        )
        return _Iterator(base, self.random)


# TODO: put it into the hydrator part when it is ok.
class ReferentialDocumentIdSamplerTransformAdapter(ReferentialSampler):

    sampler: Param[JSONLReferentialDocumentIdSampler]

    adapter: Param[SampleTransform]
    """The transformation"""

    def initialize(self, random: Optional[np.random.RandomState] = None):
        super().initialize(random)
        self.sampler.initialize(random)

    def transform_record(
        self, record: ReferentialDocumentIDRecord
    ) -> ReferentialDocumentIDRecord:
        docs = [record.document.document]
        docs = self.adapter.transform_documents(docs) or docs

        return ReferentialDocumentIDRecord(
            document=DocumentRecord(docs[0]),
            target=record.target,
        )

    def referential_iter(self) -> Iterator[ReferentialDocumentIDRecord]:
        iterator = self.sampler.referential_iter()

        return SerializableIteratorTransform(
            SkippingIterator.make_serializable(iterator), self.transform_record
        )
