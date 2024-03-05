from abc import ABC, abstractmethod
from typing import Optional, Tuple, Iterator, Any
from experimaestro import Param, Config
import torch
import numpy as np
from datamaestro_text.data.ir import DocumentStore, TextItem
from datamaestro_text.data.ir.base import (
    SimpleTextTopicRecord,
    SimpleTextDocumentRecord,
    TopicRecord,
    DocumentRecord,
    GenericTopicRecord,
    GenericDocumentRecord,
)
from xpmir.letor import Random
from xpmir.letor.records import PairwiseRecord, ProductRecords
from xpmir.letor.samplers import BatchwiseSampler, PairwiseSampler
from xpmir.utils.iter import RandomSerializableIterator, SerializableIterator


class DocumentSampler(Config, ABC):
    """How to sample from a document store"""

    documents: Param[DocumentStore]

    @abstractmethod
    def __call__(self) -> Tuple[Optional[int], Iterator[DocumentRecord]]:
        """Returns an indicative number of samples and an iterator"""
        raise NotImplementedError()

    def __iter__(self) -> Iterator[DocumentRecord]:
        """Shorthand method that directly returns an iterator"""
        _, iter = self()
        return iter


class HeadDocumentSampler(DocumentSampler):
    """A basic sampler that iterates over the first documents

    if max_count is 0, it iterates over all documents
    """

    max_count: Param[int] = 0
    """Maximum number of documents (if 0, no limit)"""

    max_ratio: Param[float] = 0
    """Maximum ratio of documents (if 0, no limit)"""

    def __call__(self) -> Tuple[int, Iterator[DocumentRecord]]:
        count = (self.max_ratio or 1) * self.documents.documentcount

        if self.max_count > 0:
            count = min(self.max_count, count)

        count = int(count)
        return count, self.iter(count)

    def iter(self, count):
        for _, document in zip(range(count), self.documents.iter_documents()):
            yield document


class RandomDocumentSampler(DocumentSampler):
    """A basic sampler that iterates over the first documents

    Either max_count or max_ratio should be non null
    """

    max_count: Param[int] = 0
    """Maximum number of documents (if 0, no limit)"""

    max_ratio: Param[float] = 0
    """Maximum ratio of documents (if 0, no limit)"""

    random: Param[Optional[Random]]
    """Random sampler"""

    def __call__(self) -> Tuple[int, Iterator[str]]:
        # Compute the number of documents to sample
        count = (self.max_ratio or 1) * self.documents.documentcount

        if self.max_count > 0:
            count = min(self.max_count, count)
        count = int(count)
        return count, self.iter(count)

    def iter(self, count) -> Iterator[str]:
        """Iterate over the documents"""
        state = np.random.RandomState() if self.random is None else self.random.state
        docids = state.choice(
            np.arange(self.documents.documentcount), size=count, replace=False
        )
        for docid in docids:
            yield self.documents.document_int(int(docid))


class RandomSpanSampler(BatchwiseSampler, PairwiseSampler):
    """This sampler uses positive samples coming from the same documents
    and negative ones coming from others

    Allows to (pre)-train as in co-condenser:
        L. Gao and J. Callan, “Unsupervised Corpus Aware Language Model
        Pre-training for Dense Passage Retrieval,” arXiv:2108.05540 [cs],
        Aug. 2021, Accessed: Sep. 17, 2021. [Online].
        http://arxiv.org/abs/2108.05540
    """

    documents: Param[DocumentStore]
    """The document store to use"""

    max_spansize: Param[int] = 1000
    """Maximum span size in number of characters"""

    def get_text_span(self, text, random):
        # return the two spans of text
        spanlen = min(self.max_spansize, len(text) // 2)

        max_start1 = len(text) - spanlen * 2
        start1 = random.randint(0, max_start1) if max_start1 > 0 else 0
        end1 = start1 + spanlen
        if start1 > 0 and text[start1 - 1] != " ":
            start1 = text.find(" ", start1) + 1
        if text[end1] != " ":
            end1 = text.rfind(" ", 0, end1)

        max_start2 = len(text) - spanlen
        start2 = random.randint(end1, max_start2) if max_start2 > end1 else end1
        end2 = start2 + spanlen
        if text[start2 - 1] != " ":
            start2 = text.find(" ", start2) + 1
        if text[end2 - 1] != " ":
            end2 = text.rfind(" ", 0, end2)

        # Rejet wrong samples
        if end2 <= start2 or end1 <= start1:
            return None

        return (text[start1:end1], text[start2:end2])

    def pairwise_iter(self) -> SerializableIterator[PairwiseRecord, Any]:
        def iter(random: np.random.RandomState):
            iter = self.documents.iter_sample(lambda m: random.randint(0, m))

            while True:
                record_pos_qry = next(iter)
                text_pos_qry = record_pos_qry[TextItem].text
                spans_pos_qry = self.get_text_span(text_pos_qry, random)

                record_neg = next(iter)
                text_neg = record_neg[TextItem].text
                spans_neg = self.get_text_span(text_neg, random)

                if not (spans_pos_qry and spans_neg):
                    continue

                yield PairwiseRecord(
                    SimpleTextTopicRecord.from_text(spans_pos_qry[0]),
                    SimpleTextDocumentRecord.from_text(spans_pos_qry[1]),
                    SimpleTextDocumentRecord.from_text(spans_neg[random.randint(0, 2)]),
                )

        return RandomSerializableIterator(self.random, iter)

    def batchwise_iter(
        self, batch_size: int
    ) -> SerializableIterator[ProductRecords, Any]:
        def iterator(random: np.random.RandomState):
            # Pre-compute relevance matrix
            relevances = torch.diag(torch.ones(batch_size, dtype=torch.float))

            iter = self.documents.iter_sample(lambda m: random.randint(0, m))

            while True:
                batch = ProductRecords()
                while len(batch) < batch_size:
                    record = next(iter)
                    text = record.text
                    res = self.get_text_span(text, random)
                    if not res:
                        continue
                    batch.add_topics(SimpleTextTopicRecord.from_text(res[0]))
                    batch.add_documents(SimpleTextDocumentRecord.from_text(res[1]))
                batch.set_relevances(relevances)
                yield batch

        return RandomSerializableIterator(self.random, iterator)


# Deprecated
class UpdatableRandomSpanSampler(RandomSpanSampler):
    """The random span sampler where the negatives are sampled in an
    hierarchical way, designed for REFERENTIAL pretraining.

    During the pretraining there is not early eos, so the number of the
    possible sequence is dimension^max_depth, and store them inside a matrix
    of dimension max_depth.

    Only support the pairwise operation for the moment
    """

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
            random_iter = self.documents.iter_sample(lambda m: random.randint(0, m))
            while True:
                # prepare the positive and the query
                record_pos_qry = next(random_iter)
                text_pos_qry = record_pos_qry.text
                spans_pos_qry = self.get_text_span(text_pos_qry, random)
                # FIXME: Assume that the ext id is transferable to int.
                id_pos_qry = int(record_pos_qry.id)

                if self.current_depth > 1:
                    # current depth = 2 so use the prefix at depth 1, so indice = 1
                    id_matrix = self.id_matrix[self.current_depth - 2]
                    res = self.hierarchical_neg_mining(id_matrix, id_pos_qry, random)
                    if res:
                        record_neg = self.documents.document_int(res)
                    else:
                        record_neg = next(random_iter)
                else:
                    record_neg = next(random_iter)

                text_neg = record_neg.text
                spans_neg = self.get_text_span(text_neg, random)
                id_neg = int(record_pos_qry.id)

                if not (spans_pos_qry and spans_neg):
                    continue

                yield PairwiseRecord(
                    TopicRecord(
                        GenericTopicRecord(id=id_pos_qry, text=spans_pos_qry[0])
                    ),
                    DocumentRecord(
                        GenericDocumentRecord(id=id_pos_qry, text=spans_pos_qry[1])
                    ),
                    DocumentRecord(
                        GenericDocumentRecord(
                            id=id_neg, text=spans_neg[random.randint(0, 2)]
                        )
                    ),
                )

        return RandomSerializableIterator(self.random, iter)
