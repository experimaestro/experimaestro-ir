from typing import Any, Optional
import torch
import numpy as np
from experimaestro import Param
from datamaestro_text.data.ir import DocumentStore

from xpmir.letor.records import PairwiseRecord, DocumentRecord
from xpmir.letor.samplers import PairwiseSampler
from xpmir.utils.iter import SerializableIterator, RandomSerializableIterator


class NegativesUpdatableSampler(PairwiseSampler):
    """The random span sampler where the negatives are sampled in an
    hierarchical way, designed for REFERENTIAL pretraining.

    During the pretraining there is not early eos, so the number of the
    possible sequence is dimension^max_depth, and store them inside a matrix
    of dimension max_depth.

    Only support the pairwise operation for the moment

    Precondition: the id of the positive o
    """

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

    id_matrix: list[torch.tensor]
    """A list of length max_depth - 1(for the last layer this matrix is
    useless), where each element is of dimension depth+1, like
    ([dim, N_1], [dim, dim, N_2], etc) which contains the id of the documents"""

    log_proba_mean_matrix: list[torch.tensor]
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
            self.id_matrix.append(
                torch.ones(shape_ids, dtype=torch.int32).detach() * (-1)
            )
            # initialize the log_probabilities with the a small probability
            self.log_proba_mean_matrix.append(torch.full(shape_mean, -10.0).detach())

    def update_matrix(
        self,
        sampled_tokens: torch.tensor,  # shape [depth, bs]
        ids: torch.tensor,  # shape: bs
        log_proba: torch.tensor,  # shape: bs
    ):
        with torch.no_grad():
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
            better_indices = torch.where(log_proba > log_proba_mean)[0]
            better_ids = ids[better_indices]
            better_sampled_tokens = sampled_tokens[:, better_indices]

            # shape [nb_better, nb_documents]
            ids_sequences = self.id_matrix[current_depth - 1][
                tuple(better_sampled_tokens)
            ]

            # randomly replace the previous ids
            indices_doc_to_replace = torch.randint(
                self.id_matrix[current_depth - 1].shape[-1],
                (ids_sequences.shape[0],),
                dtype=torch.int32,
            )

            # replace them!
            self.id_matrix[current_depth - 1][
                tuple(torch.vstack((better_sampled_tokens, indices_doc_to_replace)))
            ] = better_ids

    def hierarchical_neg_mining(
        self,
        id_matrix: torch.tensor,  # shape depends on the layer
        target_id: int,
        random: np.random.RandomState,
    ):
        """return the internal id of the negative, if not find, return None"""
        with torch.no_grad():  # this no_grad should be unneccessary
            indices = torch.where(id_matrix == target_id)
            if indices[0].shape[0] == 0:  # target id not found in matrix
                return None

            # random choose one to as the target sequence
            random_choosed = torch.vstack(indices)[
                :, random.randint(indices[0].shape[0])
            ]
            # build the hierarchical matrix to search
            # e.g. the input is at [7,8,2,6], first we search from [7,8,2],
            # if not found, we go to [7, 8], etc
            for i in range(self.current_depth - 1, 0, -1):
                hierarchical_targets = id_matrix[tuple(random_choosed[:i])]
                satisfied = torch.where(
                    torch.logical_and(
                        hierarchical_targets != -1, hierarchical_targets != target_id
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
                # FIXME: Assume that the ext id is transferable to int.
                id_pos_qry = int(original_record.positive.document.get_id())

                if self.current_depth > 1:
                    # current depth = 2 so use the prefix at depth 1, so indice = 0
                    id_matrix = self.id_matrix[self.current_depth - 2]
                    res = self.hierarchical_neg_mining(id_matrix, id_pos_qry, random)
                    if res:
                        record_neg = DocumentRecord(self.documents.document_int(res))
                    else:
                        record_neg = original_record.negative
                else:
                    record_neg = original_record.negative

                yield PairwiseRecord(
                    original_record.query, original_record.positive, record_neg
                )

        return RandomSerializableIterator(self.random, iter)
