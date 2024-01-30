from typing import Any, Optional, TypeVar
import numpy as np
from experimaestro import Param
from datamaestro_text.data.ir import DocumentStore

from xpmir.letor.records import PairwiseRecord, DocumentRecord
from xpmir.letor.samplers import PairwiseSampler
from xpmir.utils.iter import (
    SerializableIterator,
    RandomSerializableIterator,
    RandomStateSerializableAdaptor,
)

State = TypeVar("State")
T = TypeVar("T")


class NegativesUpdatableSampler(PairwiseSampler):
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

    def update_matrix(
        sampled_tokens: np.array,  # shape [depth, bs]
        ids: np.chararray,  # shape: bs
        log_proba: np.array,  # shape: bs
    ):
        pass


# FIXME: Not working for the moment, need to treat the current depth
class DepthUpdatableNegativesUpdatableSampler(NegativesUpdatableSampler):
    """The random span sampler where the negatives are sampled in an
    hierarchical way, designed for REFERENTIAL pretraining.

    During the pretraining there is not early eos, so the number of the
    possible sequence is dimension^max_depth, and store them inside a matrix
    of dimension max_depth.

    Only support the pairwise operation for the moment

    The training is based on the depth updable.
    """

    id_matrix: list[np.chararray]
    """A list of length max_depth - 1(for the last layer this matrix is
    useless), where each element is of dimension depth+1, like
    ([dim, N_1], [dim, dim, N_2], etc) which contains the id of the documents"""

    log_proba_mean_matrix: list[np.array]
    """A list of the same length of id_matrix, and each element of shape
    [dim], [dim, dim], etc... .
    It contains the log_probability of the given sequence"""

    def initialize(self, random: Optional[np.random.RandomState] = None):
        super().initialize(random)
        self.base_sampler.initialize(random)

    def __post_init__(self):
        self.id_matrix = []
        self.log_proba_mean_matrix = []

        # store the current depth
        self.current_depth = 1

        for depth in range(1, self.max_depth):
            sequence_num = self.dimension**depth
            nb_docs = int(
                self.documents.documentcount
                / sequence_num
                * self.max_docs_per_seq_coeff
            )
            shape_mean = [self.dimension for _ in range(depth)]
            shape_ids = shape_mean + [nb_docs]
            # initialize the ids with -1.
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
        random: np.random.RandomState,
    ):
        """return the internal id of the negative, if not find, return None"""
        indices = np.where(id_matrix == target_id)
        if indices[0].shape[0] == 0:  # target id not found in matrix
            return None

        # random choose one to as the target sequence
        random_choosed = np.vstack(indices)[:, random.randint(indices[0].shape[0])]
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
                return random.choice(hierarchical_targets[satisfied])

        return None  # not found over all the hierarchical matrix

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord, Any]:
        def iter(random: np.random.RandomState):
            original_iter = self.base_sampler.pairwise_iter()
            while True:
                original_record = next(original_iter)
                id_pos_qry = original_record.positive.document.get_id()

                if self.current_depth > 1:
                    # current depth = 2 so use the prefix at depth 1, so indice = 0
                    id_matrix = self.id_matrix[self.current_depth - 2]
                    res = self.hierarchical_neg_mining(id_matrix, id_pos_qry, random)
                    if res:
                        record_neg = DocumentRecord(self.documents.document_ext(res))
                    else:
                        record_neg = original_record.negative
                else:
                    record_neg = original_record.negative

                yield PairwiseRecord(
                    original_record.query, original_record.positive, record_neg
                )

        return RandomSerializableIterator(self.random, iter)


class DynamicNegativesUpdatableSampler(NegativesUpdatableSampler):
    """The random span sampler where the negatives are sampled in an
    hierarchical way, designed for REFERENTIAL pretraining.

    During the pretraining there is not early eos, so the number of the
    possible sequence is dimension^max_depth, and store them inside a matrix
    of dimension max_depth.

    Only support the pairwise operation for the moment

    Similar to the previous sampler, but only support for the full depth negative mining
    """

    id_matrix: np.chararray
    """A matrix of shape [dim, dim, ..., dim, N_2] of dim max_depth + 1, which
    contains the id of the documents"""

    log_proba_mean_matrix: np.array
    """A matrix of shape [dim, dim, ..., dim, N_2] of dim max_depth + 1, which
    contains the id of the documents
    It contains the avg log_probability of the given sequence"""

    def initialize(self, random: Optional[np.random.RandomState] = None):
        super().initialize(random)
        self.base_sampler.initialize(random)

    def __post_init__(self):
        sequence_num = self.dimension**self.max_depth
        nb_docs = int(
            self.documents.documentcount / sequence_num * self.max_docs_per_seq_coeff
        )

        shape_proba = [self.dimension for _ in range(self.max_depth)]
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

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord, Any]:
        class _Iterator(
            RandomStateSerializableAdaptor[SerializableIterator[PairwiseRecord]]
        ):
            def __init__(
                self,
                iterator: SerializableIterator[PairwiseRecord],
                id_matrix,
                id_proba,
                documents,
                max_depth,
            ):
                super().__init__(iterator)
                self.id_matrix = id_matrix
                self.id_proba = id_proba
                self.documents = documents
                self.max_depth = max_depth

            def load_state_dict(self, state: Any):
                # state = iterator state + negative states.
                self.id_matrix = state[0]["id_matrix"]
                self.id_proba = state[0]["id_proba"]
                self.iterator.load_state_dict(state[1])

            def state_dict(self) -> Any:
                iterator_state = self.iterator.state_dict()
                negative_state = {
                    "id_matrix": self.id_matrix,
                    "id_proba": self.id_proba,
                }
                return (negative_state, iterator_state)

            def hierarchical_neg_mining(
                self,
                id_matrix: np.chararray,  # shape depends on the layer
                target_id: str,  # the string id
            ):
                """return the internal id of the negative, if not find, return None"""
                indices = np.where(id_matrix == target_id)
                if indices[0].shape[0] == 0:  # target id not found in matrix
                    return None

                # random choose one to as the target sequence
                random_choosed = np.vstack(indices)[
                    :, self.random.randint(indices[0].shape[0])
                ]
                # build the hierarchical matrix to search
                # e.g. the input is at [7,8,2,6], first we search from [7,8,2],
                # if not found, we go to [7, 8], etc
                for i in range(self.max_depth, 0, -1):
                    hierarchical_targets = id_matrix[tuple(random_choosed[:i])]
                    satisfied = np.where(
                        np.logical_and(
                            hierarchical_targets != "",
                            hierarchical_targets != target_id,
                        )
                    )
                    if (
                        hierarchical_targets[satisfied].shape[0] > 0
                    ):  # found at this level
                        return self.random.choice(hierarchical_targets[satisfied])

                return None  # not found over all the hierarchical matrix

            def __next__(self) -> SerializableIterator[PairwiseRecord, Any]:
                original_pair = next(self.iterator)
                id_pos_qry = original_pair.positive.document.get_id()
                res = self.hierarchical_neg_mining(self.id_matrix, id_pos_qry)
                if res:
                    record_neg = DocumentRecord(self.documents.document_ext(res))
                else:
                    record_neg = original_pair.negative

                return PairwiseRecord(
                    original_pair.query, original_pair.positive, record_neg
                )

        return _Iterator(
            self.base_sampler.pairwise_iter(),
            self.id_matrix,
            self.log_proba_mean_matrix,
            self.documents,
            self.max_depth,
        )
