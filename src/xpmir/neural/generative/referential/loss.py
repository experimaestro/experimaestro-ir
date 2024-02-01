import logging
from dataclasses import dataclass
from typing import List, Optional, NamedTuple, Generic, TypeVar, Callable
from datamaestro_text.data.ir import DocumentStore

import torch
import numpy as np
from experimaestro import Param
from experimaestro.compat import cached_property
from torch import nn

from xpmir.learning.context import Loss
from xpmir.learning.metrics import ScalarMetric
from xpmir.letor.records import BaseRecords, PairwiseRecords
from xpmir.letor.trainers import TrainerContext
from xpmir.letor.trainers.generative import PairwiseGenerativeLoss
from xpmir.neural.generative import (
    ConditionalGenerator,
    StepwiseGenerator,
)
from xpmir.utils.utils import easylog
from xpmir.context import Hook
from xpmir.neural.generative.referential import DepthUpdatable
from xpmir.neural.generative.referential.samplers import DynamicNegativesSampler

logger = easylog()
T = TypeVar("T")


# dataclass for training, compose of pos_doc, neg_doc and query
@dataclass
class Triplet(Generic[T]):
    pos_doc: T
    neg_doc: T
    query: T

    def __iter__(self):
        yield self.pos_doc
        yield self.neg_doc
        yield self.query


# dataclass for a pair of positive and query, used for in-loss negative sampling
@dataclass
class QryDocPair(Generic[T]):
    pos_doc: T
    query: T

    def __iter__(self):
        yield self.pos_doc
        yield self.query


class GenerativeLossOutput(NamedTuple):
    recursive_loss: torch.Tensor
    """The recursive loss"""

    kl_div_loss: torch.Tensor
    """The kl_div_loss"""

    pairwise_accuracy: torch.Tensor
    """The pairwise accuracy"""

    sampled_tokens: Optional[torch.Tensor] = None
    """The sampled_tokens if needed, of shape [depth, bs]"""

    log_current_node_proba: Optional[Triplet[torch.Tensor]] = None
    """The probability of the current node, shape [bs]"""

    sampled_tokens_proba: Optional[Triplet[torch.Tensor]] = None
    """The probability of the sampled tokens, shape [current_max_depth]"""


class GenerativeQryDocPreparation(NamedTuple):
    """A dataclass to store the pre-calculated tokens and probas"""

    sampled_tokens: Optional[torch.Tensor] = None
    """The sampled_tokens if needed, of shape [depth, bs]"""

    log_proba: Optional[QryDocPair[torch.Tensor]] = None
    """The log_proba at each depth, of shape [depth, bs, dec_dim+1]"""


# -- Hooks
class LossProcessHook(Hook):
    def process(self):
        """Called after process"""
        pass


class NegativeUpdateHook(LossProcessHook):

    sampler: Param[DynamicNegativesSampler]
    """The sampler which contains the hard negatives"""

    max_depth: Param[int]
    """The max depth for the model"""

    def process(
        self,
        loss_output: GenerativeLossOutput,
        records: BaseRecords,
        current_max_depth: int,
    ):
        ext_ids = [pdr.document.get_id() for pdr in records.positives]
        # get the tokens
        sampled_tokens = (
            loss_output.sampled_tokens.cpu().detach().numpy()
        )  # shape [depth, bs]
        # get the log_proba, shape [bs]
        log_proba = loss_output.log_current_node_proba.pos_doc.cpu().detach().numpy()

        self.sampler.pairwise_iter().update_matrix(
            sampled_tokens,  # shape [depth, bs]
            np.array(ext_ids, dtype="<U10"),  # shape [bs]
            log_proba,  # shape [bs]
        )


# Dynamic negative holder
class DynamicNegativeBuilder:
    """A class to store the id_matrix and correpond proba benchmark,
    also provide a method to update it

    Only support the fix max_depth for the moment
    """

    id_matrix: np.chararray
    """A matrix of shape [dim, dim, ..., dim, N_2] of dim max_depth + 1, which
    contains the id of the documents"""

    log_proba_mean_matrix: np.ndarray
    """A matrix of shape [dim, dim, ..., dim, N_2] of dim max_depth + 1, which
    contains the id of the documents
    It contains the avg log_probability of the given sequence"""

    def __init__(
        self,
        documents: DocumentStore,
        max_depth: int,
        dimension: int,
        max_docs_per_seq_coeff: float = 1.5,
        fake_average_coeff: float = 0.8,
    ):
        self.documents = documents
        self.max_depth = max_depth
        self.fake_average_coeff = fake_average_coeff
        self.dimension = dimension
        sequence_num = self.dimension**self.max_depth
        nb_docs = int(
            self.documents.documentcount / sequence_num * max_docs_per_seq_coeff
        )

        shape_proba = [self.dimension for _ in range(self.max_depth)]
        shape_ids = shape_proba + [nb_docs]
        # initialize the ids with -1.
        self.id_matrix = np.empty(shape_ids, dtype="<U10")
        # initialize the log_probabilities with the a small probability
        self.log_proba_mean_matrix = np.full(shape_proba, -10.0)

    def update_matrix(
        self,
        sampled_tokens: np.ndarray,  # shape [max_depth, bs]
        ids: np.chararray,  # shape: bs
        log_proba: np.ndarray,  # shape: bs
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

    def hard_negative_mining(
        self,
        sampled_tokens: np.ndarray,  # shape [depth, bs]
        negative_level: int,  # 0 represent the random, or layer of negatives
        positive_ids: np.ndarray,  # shape [bs],
        document_prefix: str,
    ):
        """return a list of length bs, if empty means this position don't have
        have hard negatives, so still use random negatives"""
        bs = sampled_tokens.shape[1]
        negative_text_list = ["" for _ in range(bs)]
        if negative_level == 0:
            return negative_text_list

        fixed_prefix = list(sampled_tokens[:negative_level])
        # append the random sampled prefix
        for _ in range(self.max_depth - negative_level):
            fixed_prefix.append(np.random.randint(self.dimension, size=bs))
        fixed_prefix.append(np.random.randint(self.id_matrix.shape[-1], size=bs))

        # get the corresponding ids
        ids = self.id_matrix[fixed_prefix]
        # filter the ids to remove one with the same id as positive and ''
        filtered_indice = np.where(np.logical_and(ids != "", ids != positive_ids))
        negative_texts = self.documents.documents_ext(list(ids[filtered_indice]))

        for i, batch_indice in enumerate(filtered_indice):
            negative_text_list[batch_indice] = document_prefix + negative_texts[i]
        return negative_text_list


# --- For training
class PairwiseGenerativeRetrievalLoss(PairwiseGenerativeLoss, DepthUpdatable):
    NAME = "PairwiseGenerativeLoss"

    documents: Param[DocumentStore]
    """The document store"""

    id_generator: Param[ConditionalGenerator]
    """The id generator"""

    alpha: Param[float] = 0.0
    """The hyperparameter for the KL divergence"""

    loss_hooks: Param[List[LossProcessHook]] = []
    """The hook"""

    dynamic_negatives: Param[bool] = False
    """Whether build the dynamic negatives inside the loss part"""

    document_prefix: Param[str] = ""
    """The document prefix for the negatives"""

    @cached_property
    def kl_target(self):
        assert self.id_generator.eos_token_id == self.id_generator.decoder_outdim
        decoder_outdim = self.id_generator.decoder_outdim
        alphas = torch.tensor(
            [
                sum(decoder_outdim**i for i in range(j + 1))
                for j in range(self.max_depth, 0, -1)
            ]
        ).to(self.id_generator.device)
        alphas = (1 / alphas).unsqueeze(-1)
        return torch.log(
            torch.cat(
                (((1 - alphas) / decoder_outdim).expand(-1, decoder_outdim), alphas), -1
            )
        )

    @cached_property
    def perfect_proba(self):
        max_depth = self.kl_target.shape[0]
        probas = []
        for stop_recursive_flag in range(max_depth):
            probas.append(self.perfect_proba_recursive(0, stop_recursive_flag))
        return torch.tensor(probas).to(self.id_generator.device)

    def perfect_proba_recursive(self, current_depth, stop_recursive_flag):
        k = self.id_generator.decoder_outdim
        other_proba = torch.exp(
            self.kl_target[current_depth, 0]
        )  # any token which is not eos
        alpha = torch.exp(
            self.kl_target[current_depth, self.id_generator.eos_token_id]
        )  # eos
        current_layer_value = k * other_proba * (1 - other_proba) + alpha * (1 - alpha)
        if current_depth == self.max_depth - 1 or current_depth == stop_recursive_flag:
            return current_layer_value

        return (
            self.perfect_proba_recursive(current_depth + 1, stop_recursive_flag)
            * k
            * other_proba**2
            + current_layer_value
        )

    def initialize(self):
        self.kl_lossfn = nn.KLDivLoss(reduction="sum", log_target=True)
        if self.dynamic_negatives:
            logger.info("Build the instance for dynamic negative sampling")
            self.dynamic_negatives_builder = DynamicNegativeBuilder(
                documents=self.documents,
                max_depth=self.current_max_depth,
                dimension=self.id_generator.decoder_outdim,
            )

    def prepare_sampling_target(
        self, bs
    ) -> Callable[[Triplet[torch.Tensor]], torch.Tensor]:
        """Return a function that take the log_proba triplets and return the
        proba to be sampled"""

        def log_p_next_token(log_proba: Triplet[torch.Tensor]):
            return torch.stack(
                [
                    log_proba.query[:, :-1] * 0.5,
                    log_proba.pos_doc[:, :-1] * 0.5,
                ]
            ).logsumexp(
                dim=0
            )  # shape [bs, decoder_dim]

        return log_p_next_token

    def recursive(
        self,
        decoder_input_tokens,  # None or [depth - 1, bs], or [current_max_depth, bs]
        unfinished_sequences: torch.Tensor,  # shape [bs]
        log_cur_node_proba: Triplet[torch.Tensor],  # shape [bs,3]
        depth: int,  # a value counting from 1
        stepwise_generators: Triplet[StepwiseGenerator],
        log_p_next_token_generator: Callable[[Triplet[torch.Tensor]], torch.Tensor],
        previous_sampled_token_p: Triplet[torch.Tensor],
        cached_log_proba: Optional[
            QryDocPair[torch.Tensor]
        ] = None,  # shape [depth, bs, dec_dim+1]
    ) -> GenerativeLossOutput:
        """Return the recursive G and the kl_div loss at depth, also the
        pairwise accruracy for supervision"""
        assert depth <= self.current_max_depth
        if cached_log_proba is None:
            # pass get the probas, each one of shape: [bs, dec_dim+1]
            if decoder_input_tokens is None:
                logits = Triplet(
                    *(g.step(decoder_input_tokens) for g in stepwise_generators)
                )
            else:
                logits = Triplet(
                    *(g.step(decoder_input_tokens[-1]) for g in stepwise_generators)
                )

            log_proba = Triplet(
                *(nn.functional.log_softmax(logit, dim=-1) for logit in logits)
            )

            # clamp the log_probas to avoid nan in the loss
            log_proba = Triplet(*(torch.clamp_min(log_p, -14) for log_p in log_proba))
        else:
            # get the log proba at the corresponding depth
            if depth == 1:
                neg_doc_logits = stepwise_generators.neg_doc.step(None)
            else:
                # when depth = 2, we use the decoder input at depth 1 so indice 0
                neg_doc_logits = decoder_input_tokens[depth - 2]
            log_proba = Triplet(
                pos_doc=cached_log_proba.pos_doc[depth - 1],
                neg_doc=torch.clamp_min(
                    nn.functional.log_softmax(neg_doc_logits, dim=-1), -14
                ),
                query=cached_log_proba.query[depth - 1],
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("\n")
            logger.debug(f"posdoc_proba: {torch.exp(log_proba.pos_doc)}")
            logger.debug(f"negdoc_proba: {torch.exp(log_proba.neg_doc)}")
            logger.debug(f"query_proba: {torch.exp(log_proba.query)}")

        # --- middle_term in the loss formula
        middle_term = torch.sum(
            torch.exp(log_proba.pos_doc.detach() + log_proba.query.detach())
            * (1 - torch.exp(log_proba.neg_doc.detach()))
            * (
                log_cur_node_proba.pos_doc.unsqueeze(-1)
                + log_cur_node_proba.neg_doc.unsqueeze(-1)
                + log_cur_node_proba.query.unsqueeze(-1)
                + log_proba.pos_doc
                + log_proba.query
            ),
            dim=-1,
        )  # shape: [bs]

        # --- last term in the loss formula
        max_values_tmp = torch.max(
            log_proba.pos_doc.detach() + log_proba.query.detach(), dim=-1
        ).values
        sum_except_current = (
            torch.sum(
                torch.exp(
                    log_proba.pos_doc.detach()
                    + log_proba.query.detach()
                    - max_values_tmp.unsqueeze(-1)
                ),
                dim=-1,
            )
            * torch.exp(max_values_tmp)
        ).unsqueeze(-1) - torch.exp(
            log_proba.pos_doc.detach() + log_proba.query.detach()
        )
        last_term = torch.sum(
            torch.exp(log_proba.neg_doc.detach())
            * sum_except_current
            * log_proba.neg_doc,
            dim=-1,
        )  # shape: [bs]

        # --- last term for the pairwise accuracy formula
        with torch.no_grad():
            pairwise_accuracy_middle_term = torch.sum(
                torch.exp(log_proba.pos_doc + log_proba.query)
                * (1 - torch.exp(log_proba.neg_doc)),
                dim=-1,
            )

        # --- Sample the next token

        # currently only support the eos_token_id is the last one the decoding dimension
        assert self.id_generator.eos_token_id == log_proba.pos_doc.shape[1] - 1

        # get the bs
        bs = log_cur_node_proba.query.shape[0]

        # log-probability of the next token using a mixture
        with torch.no_grad():
            if (
                decoder_input_tokens is None
                or decoder_input_tokens.shape[0] != self.current_max_depth
            ):
                log_p_next_token = log_p_next_token_generator(log_proba)
                next_tokens = torch.multinomial(
                    torch.exp(log_p_next_token), num_samples=1
                ).squeeze(
                    1
                )  # shape [bs]

                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"sampled token is {next_tokens} of depth {depth}")

                # append the tokens to the next_step to make it to shape: [depth, bs]
                if decoder_input_tokens is None:
                    decoder_input_tokens = next_tokens.unsqueeze(0)
                else:
                    decoder_input_tokens = torch.vstack(
                        (decoder_input_tokens, next_tokens)
                    )
            else:
                log_p_next_token = log_p_next_token_generator(log_proba)
                next_tokens = decoder_input_tokens[depth - 1]

        # --- Computes the log probability of sampled tokens

        batch_range = torch.arange(bs)
        # each of shape [bs]
        log_proba_next = Triplet(*(x[batch_range, next_tokens] for x in log_proba))
        log_cur_node_proba = Triplet(
            *(a + b for a, b in zip(log_cur_node_proba, log_proba_next))
        )

        # mask the generated tokens if some of the seqs
        # are already end before(0 in unfinished_sequences)
        new_unfinished_sequences = (
            next_tokens != self.id_generator.eos_token_id
        ) & unfinished_sequences

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"input token for the next step: {next_tokens}")

        # the kl loss to force all the sequence to have the similar proba and we
        # mask out the early finished ones
        kl_loss = Triplet(
            *(
                self.kl_lossfn(
                    torch.log(
                        torch.mean(
                            torch.exp(x[unfinished_sequences.type(torch.bool)]), 0
                        )
                    ),
                    self.kl_target[depth - 1],
                )
                for x in log_proba
            )
        )
        kl_loss = kl_loss.pos_doc + kl_loss.neg_doc + kl_loss.query

        # prepare the sampled probability for logging
        with torch.no_grad():
            sampled_token_p = Triplet(
                *(
                    torch.mean(
                        torch.exp(log_pro[batch_range, next_tokens]),
                        dim=0,
                        keepdim=True,
                    )
                    for log_pro in log_proba
                )
            )

            if previous_sampled_token_p:
                sampled_token_p = Triplet(
                    *(
                        torch.cat((previous, current))
                        for previous, current in zip(
                            previous_sampled_token_p, sampled_token_p
                        )
                    )
                )

        # whether need to be end now?
        if new_unfinished_sequences.max() == 0 or depth == self.current_max_depth:
            return GenerativeLossOutput(
                recursive_loss=(middle_term + last_term)
                * unfinished_sequences.detach(),
                kl_div_loss=kl_loss,
                pairwise_accuracy=pairwise_accuracy_middle_term
                * unfinished_sequences.detach(),
                sampled_tokens=decoder_input_tokens,
                log_current_node_proba=log_cur_node_proba,
                sampled_tokens_proba=sampled_token_p,
            )

        # Computing the importance sampling coefficient
        with torch.no_grad():
            log_p = log_p_next_token[batch_range, next_tokens]
            sampling_multiplier = torch.exp(
                log_proba_next.pos_doc
                + log_proba_next.neg_doc
                + log_proba_next.query
                - log_p
            )

        next_layer_recursive = self.recursive(
            decoder_input_tokens,
            new_unfinished_sequences,
            log_cur_node_proba,
            depth + 1,
            stepwise_generators,
            log_p_next_token_generator,
            sampled_token_p,
        )

        return GenerativeLossOutput(
            recursive_loss=unfinished_sequences.detach()
            * (
                next_layer_recursive.recursive_loss * sampling_multiplier
                + middle_term
                + last_term
            ),
            kl_div_loss=next_layer_recursive.kl_div_loss + kl_loss,
            pairwise_accuracy=unfinished_sequences.detach()
            * (
                next_layer_recursive.pairwise_accuracy * sampling_multiplier
                + pairwise_accuracy_middle_term
            ),
            sampled_tokens=next_layer_recursive.sampled_tokens,
            log_current_node_proba=next_layer_recursive.log_current_node_proba,
            sampled_tokens_proba=next_layer_recursive.sampled_tokens_proba,
        )

    def query_document_pre_recursive(
        self,
        depth: int,  # start from 1
        stepwise_generators: QryDocPair[StepwiseGenerator],
        log_p_next_token_generator: Callable[[Triplet[torch.Tensor]], torch.Tensor],
        log_proba: Optional[QryDocPair[torch.Tensor]],  # shape [depth-1, bs, dec_dim+1]
        decoder_input_tokens: Optional[torch.Tensor],  # none, or shape [depth-1, bs]
    ) -> GenerativeQryDocPreparation:
        assert depth <= self.current_max_depth
        if decoder_input_tokens is None:
            logits = QryDocPair(
                *(g.step(decoder_input_tokens) for g in stepwise_generators)
            )
        else:
            logits = QryDocPair(
                *(g.step(decoder_input_tokens[-1]) for g in stepwise_generators)
            )

        new_log_proba = QryDocPair(
            *(nn.functional.log_softmax(logit, dim=-1) for logit in logits)
        )  # shape [bs, dec_dim + 1]

        # clamp the log_probas to avoid nan in the loss
        new_log_proba = QryDocPair(
            *(torch.clamp_min(log_p, -14) for log_p in new_log_proba)
        )

        # append the new_log_proba for real recursive
        if log_proba is None:
            log_proba = QryDocPair(
                *(new_log_p.unsqueeze(0) for new_log_p in new_log_proba)
            )
        else:
            log_proba = QryDocPair(
                *(
                    torch.vstack((log_p, new_log_p.unsqueeze(0)))
                    for log_p, new_log_p in zip(log_proba, new_log_proba)
                )
            )  # shape [depth, bs, dec_dim+1]

        # currently only support the eos_token_id is the last one the decoding dimension
        assert self.id_generator.eos_token_id == new_log_proba.pos_doc.shape[1] - 1

        with torch.no_grad():
            # although here the log_p_next_token_generator allow for Triplet, it
            # can also receive the QryDocPair if the negative samples are not
            # involved
            log_p_next_token = log_p_next_token_generator(new_log_proba)
            next_tokens = torch.multinomial(
                torch.exp(log_p_next_token), num_samples=1
            ).squeeze(
                1
            )  # shape [bs]

            # append the tokens to the next_step to make it to shape: [depth, bs]
            if decoder_input_tokens is None:
                decoder_input_tokens = next_tokens.unsqueeze(0)
            else:
                decoder_input_tokens = torch.vstack((decoder_input_tokens, next_tokens))

        # whether need to be end now?
        if depth == self.current_max_depth:
            return GenerativeQryDocPreparation(
                sampled_tokens=decoder_input_tokens,
                log_proba=log_proba,
            )

        return self.query_document_pre_recursive(
            depth + 1,
            stepwise_generators,
            log_p_next_token_generator,
            log_proba,
            decoder_input_tokens,
        )

    def compute(
        self, records: PairwiseRecords, context: TrainerContext
    ) -> GenerativeLossOutput:
        posdocs_text = [pdr.document.get_text() for pdr in records.positives]
        queries_text = [qr.topic.get_text() for qr in records.unique_queries]
        negdocs_text = [ndr.document.get_text() for ndr in records.negatives]
        bs = len(posdocs_text)
        # prepare the sampling target
        log_p_next_token_generator = self.prepare_sampling_target(bs)
        if self.dynamic_negatives:
            stepwise_generators = QryDocPair(
                *(self.id_generator.stepwise_iterator() for _ in range(2))
            )
            stepwise_generators.pos_doc.init(posdocs_text)
            stepwise_generators.query.init(queries_text)
            preparation_output = self.query_document_pre_recursive(
                depth=1,
                stepwise_generators=stepwise_generators,
                log_p_next_token_generator=log_p_next_token_generator,
                log_proba=None,
                decoder_input_tokens=None,
            )

            # from the based on the sampled tokens to mine the negatives
            # get the positive document ids to
            # avoid sampling the same document as negatives
            posdocs_ids = [pdr.document.get_id() for pdr in records.positives]
            negative_level = np.random.randint(self.current_max_depth + 1)
            hard_negative_text = self.dynamic_negatives_builder.hard_negative_mining(
                sampled_tokens=preparation_output.sampled_tokens.cpu().detach().numpy(),
                negative_level=negative_level,
                positive_ids=np.array(posdocs_ids),
                document_prefix=self.document_prefix,
            )

            for indice in range(bs):
                if hard_negative_text[indice] == "":
                    hard_negative_text[indice] = negdocs_text[indice]

            # some initializations
            sampled_token_p = None  # the proba for the sampled tokens, for logging
            stepwise_generators = Triplet(  # not need for posdoc and query
                neg_doc=self.id_generator.stepwise_iterator(), pos_doc=None, query=None
            )

            unfinished_sequences = torch.ones(bs, dtype=torch.long).to(
                self.id_generator.device
            )
            log_cur_node_proba = Triplet(
                *(torch.zeros(bs).to(self.id_generator.device) for _ in range(3))
            )
            loss = self.recursive(
                preparation_output.sampled_tokens,  # shape [max_depth, bs]
                unfinished_sequences,
                log_cur_node_proba,  # Triplet[torch.Tensor], shape bs
                1,
                stepwise_generators,
                log_p_next_token_generator,  # a callable
                sampled_token_p,
                preparation_output.log_proba,
            )
        else:
            # create the generator for the given records,
            # represent the posdoc, negdoc, query, respectively
            stepwise_generators = Triplet(
                *(self.id_generator.stepwise_iterator() for _ in range(3))
            )
            stepwise_generators.pos_doc.init(posdocs_text)
            stepwise_generators.neg_doc.init(negdocs_text)
            stepwise_generators.query.init(queries_text)

            # initialize cumulate product of from the root to the current one
            log_cur_node_proba = Triplet(
                *(torch.zeros(bs).to(self.id_generator.device) for _ in range(3))
            )

            # initialize the input for the decoder and the mask
            decoder_input_tokens = None
            sampled_token_p = None  # the proba for the sampled tokens, for logging
            unfinished_sequences = torch.ones(bs, dtype=torch.long).to(
                self.id_generator.device
            )

            # prepare the sampling target
            log_p_next_token_generator = self.prepare_sampling_target(bs)

            # in fact, we need to minus something to get the pure gradient, but at
            # the level of the root, the additional terms always equals to 0
            loss = self.recursive(
                decoder_input_tokens,
                unfinished_sequences,
                log_cur_node_proba,
                1,
                stepwise_generators,
                log_p_next_token_generator,
                sampled_token_p,
            )

        return GenerativeLossOutput(
            recursive_loss=-torch.mean(loss.recursive_loss),
            kl_div_loss=loss.kl_div_loss,
            pairwise_accuracy=torch.mean(loss.pairwise_accuracy),
            sampled_tokens=loss.sampled_tokens,
            log_current_node_proba=loss.log_current_node_proba,
            sampled_tokens_proba=loss.sampled_tokens_proba,
        )

    def process(self, records: BaseRecords, context: TrainerContext):
        loss_output = self.compute(records, context)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Recursive Loss: {loss_output.recursive_loss}")
            logger.debug(f"KL_div regularization: {loss_output.kl_div_loss}")
        context.add_loss(
            Loss("recursive-loss", loss_output.recursive_loss, self.weight)
        )
        context.add_loss(Loss("kl-div-loss", loss_output.kl_div_loss, self.alpha))
        context.add_metric(
            ScalarMetric(
                "Pairwise-accuracy", float(loss_output.pairwise_accuracy), len(records)
            )
        )
        context.add_metric(
            ScalarMetric(
                "Current-accuracy / Perfect-accuracy",
                float(loss_output.pairwise_accuracy)
                / float(self.perfect_proba[self.current_max_depth - 1]),
                len(records),
            )
        )

        for depth in range(1, self.current_max_depth + 1):
            context.add_metric(
                ScalarMetric(
                    f"Sampled proba for posdoc at depth {depth}",
                    float(loss_output.sampled_tokens_proba.pos_doc[depth - 1]),
                    len(records),
                )
            )
            context.add_metric(
                ScalarMetric(
                    f"Sampled proba for negdoc at depth {depth}",
                    float(loss_output.sampled_tokens_proba.neg_doc[depth - 1]),
                    len(records),
                )
            )
            context.add_metric(
                ScalarMetric(
                    f"Sampled proba for query at depth {depth}",
                    float(loss_output.sampled_tokens_proba.query[depth - 1]),
                    len(records),
                )
            )

        if self.dynamic_negatives:
            ext_ids = [pdr.document.get_id() for pdr in records.positives]
            # get the tokens
            sampled_tokens = (
                loss_output.sampled_tokens.cpu().detach().numpy()
            )  # shape [depth, bs]
            log_proba = (
                loss_output.log_current_node_proba.pos_doc.cpu().detach().numpy()
            )
            self.dynamic_negatives_builder.update_matrix(
                sampled_tokens,  # shape [depth, bs]
                np.array(ext_ids, dtype="<U10"),  # shape [bs]
                log_proba,  # shape [bs]
            )


class BatchBasedSamplingPairwiseGenerativeRetrievalLoss(
    PairwiseGenerativeRetrievalLoss
):
    """In this loss, we sampling over query or positive documents.
    In one batch for different depth, the sampling target will not change"""

    def prepare_sampling_target(
        self, bs
    ) -> Callable[[Triplet[torch.Tensor]], torch.Tensor]:
        posdoc = torch.randint(2, (bs,)).unsqueeze(1).to(self.id_generator.device)
        posdoc = posdoc.expand(bs, self.id_generator.decoder_outdim)

        def log_p_next_token(log_proba: Triplet[torch.Tensor]):
            nonlocal posdoc
            return torch.where(
                posdoc == 1, log_proba.pos_doc[:, :-1], log_proba.query[:, :-1]
            )

        return log_p_next_token


class RandomSamplingPairwiseGenerativeRetrievalLoss(PairwiseGenerativeRetrievalLoss):
    """In this loss, we sampling over a random coefficient between query and
    document
    In one batch for different depth, the sampling target will not change"""

    def prepare_sampling_target(
        self, bs
    ) -> Callable[[Triplet[torch.Tensor]], torch.Tensor]:
        posdoc = torch.rand((bs,)).to(self.id_generator.device)

        def log_p_next_token(log_proba: Triplet[torch.Tensor]):
            nonlocal posdoc
            query = 1 - posdoc
            return torch.stack(
                [
                    log_proba.query[:, :-1] * posdoc.unsqueeze(-1),
                    log_proba.pos_doc[:, :-1] * query.unsqueeze(-1),
                ]
            ).logsumexp(
                dim=0
            )  # shape [bs, decoder_dim]

        return log_p_next_token
