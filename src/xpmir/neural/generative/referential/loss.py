import logging
from dataclasses import dataclass
from typing import List, Optional, NamedTuple, Generic, TypeVar

import torch
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
from xpmir.neural.generative.referential.samplers import NegativesUpdatableSampler

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


class GenerativeLossOutput(NamedTuple):
    recursive_loss: torch.tensor
    """The recursive loss"""

    kl_div_loss: torch.tensor
    """The kl_div_loss"""

    pairwise_accuracy: torch.tensor
    """The pairwise accuracy"""

    sampled_tokens: Optional[torch.tensor] = None
    """The sampled_tokens if needed, of shape [depth, bs]"""

    log_current_node_proba: Optional[Triplet[torch.tensor]] = None
    """The probability of the current node, shape [bs]"""


# -- Hooks
class LossProcessHook(Hook):
    def process(self):
        """Called after process"""
        pass


class NegativeUpdateHook(LossProcessHook):

    sampler: Param[NegativesUpdatableSampler]
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
        # FIXME: Transform the external ids to internal ones
        int_ids = [int(ext_id) for ext_id in ext_ids]
        self.sampler.current_depth = current_max_depth
        if current_max_depth < self.max_depth:
            # we are not at the final depth, need to update matrix
            # get the tokens
            sampled_tokens = loss_output.sampled_tokens.cpu()  # shape [depth, bs]

            # get the log_probas, shape [bs]
            log_proba = loss_output.log_current_node_proba
            # we use a max pooling for the pos and qry as in the pretraining
            # they belong to the same document
            log_proba_pos = torch.max(log_proba.query, log_proba.pos_doc).cpu()

            self.sampler.update_matrix(
                sampled_tokens,  # shape [depth, bs]
                torch.tensor(int_ids, dtype=torch.int32),  # shape [bs]
                log_proba_pos,  # shape [bs]
            )


# --- For training
class PairwiseGenerativeRetrievalLoss(PairwiseGenerativeLoss, DepthUpdatable):
    NAME = "PairwiseGenerativeLoss"

    id_generator: Param[ConditionalGenerator]
    """The id generator"""

    alpha: Param[float] = 0.0
    """The hyperparameter for the KL divergence"""

    loss_hooks: Param[List[LossProcessHook]] = []
    """The hook"""

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

    def prepare_sampling_target(self, bs):
        """The default sampling target: sampling from the conjoint of posdoc and
        query"""
        return Triplet(
            pos_doc=torch.ones(bs).to(self.id_generator.device) * 0.5,
            neg_doc=torch.full((bs,), float("inf")).to(self.id_generator.device),
            query=torch.ones(bs).to(self.id_generator.device) * 0.5,
        )

    def log_p_next_token(
        self, log_probas: Triplet, sampling_target: Triplet[torch.Tensor]
    ):
        log_p_next_token = torch.stack(
            [
                log_proba[:, :-1] * coeff.unsqueeze(-1)
                for coeff, log_proba in zip(sampling_target, log_probas)
            ]
        ).logsumexp(
            dim=0
        )  # shape [bs, decoder_dim]

        return log_p_next_token

    def recursive(
        self,
        decoder_input_tokens,  # None or [depth - 1, bs]
        unfinished_sequences: torch.tensor,  # shape [bs]
        log_cur_node_proba: Triplet[torch.Tensor],  # shape [bs,3]
        depth: int,  # a value counting from 1
        stepwise_generators: Triplet[StepwiseGenerator],
        sampling_target: Triplet[torch.Tensor],
    ) -> GenerativeLossOutput:
        """Return the recursive G and the kl_div loss at depth, also the
        pairwise accruracy for supervision"""
        assert depth <= self.current_max_depth
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
            log_p_next_token = self.log_p_next_token(log_proba, sampling_target)
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
                decoder_input_tokens = torch.vstack((decoder_input_tokens, next_tokens))

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
            sampling_target,
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
        )

    def compute(
        self, records: PairwiseRecords, context: TrainerContext
    ) -> GenerativeLossOutput:
        posdocs_text = [pdr.document.get_text() for pdr in records.positives]
        negdocs_text = [ndr.document.get_text() for ndr in records.negatives]
        queries_text = [qr.topic.get_text() for qr in records.unique_queries]

        bs = len(posdocs_text)

        logger.debug("posdocs_text: %s", posdocs_text)
        logger.debug("negdocs_text: %s", negdocs_text)
        logger.debug("queries_text: %s", queries_text)

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
        unfinished_sequences = torch.ones(bs, dtype=torch.long).to(
            self.id_generator.device
        )

        # prepare the sampling target
        sampling_target = self.prepare_sampling_target(bs)

        # in fact, we need to minus something to get the pure gradient, but at
        # the level of the root, the additional terms always equals to 0
        loss = self.recursive(
            decoder_input_tokens,
            unfinished_sequences,
            log_cur_node_proba,
            1,
            stepwise_generators,
            sampling_target,
        )

        return GenerativeLossOutput(
            recursive_loss=-torch.mean(loss.recursive_loss),
            kl_div_loss=loss.kl_div_loss,
            pairwise_accuracy=torch.mean(loss.pairwise_accuracy),
            sampled_tokens=loss.sampled_tokens,
            log_current_node_proba=loss.log_current_node_proba,
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

        for hook in self.loss_hooks:
            hook.process(loss_output, records, self.current_max_depth)


class BatchBasedSamplingPairwiseGenerativeRetrievalLoss(
    PairwiseGenerativeRetrievalLoss
):
    """In this loss, we sampling over query or positive documents.
    In one batch for different depth, the sampling target will not change"""

    def prepare_sampling_target(self, bs):
        if float(torch.rand(1)) > 0.5:
            logger.debug("sampling over the positive document")
            return Triplet(
                pos_doc=torch.ones(bs).to(self.id_generator.device),
                neg_doc=torch.full((bs,), float("inf")).to(self.id_generator.device),
                query=torch.full((bs,), float("inf")).to(self.id_generator.device),
            )
        else:
            logger.debug("sampling over the query")
            return Triplet(
                pos_doc=torch.full((bs,), float("inf")).to(self.id_generator.device),
                neg_doc=torch.full((bs,), float("inf")).to(self.id_generator.device),
                query=torch.ones(bs).to(self.id_generator.device),
            )


class RandomSamplingPairwiseGenerativeRetrievalLoss(PairwiseGenerativeRetrievalLoss):
    """In this loss, we sampling over a random coefficient between query and
    document
    In one batch for different depth, the sampling target will not change"""

    def prepare_sampling_target(self, bs):
        posdoc_sampling_target = torch.rand((bs,)).to(self.id_generator.device)
        return Triplet(
            pos_doc=posdoc_sampling_target,
            neg_doc=torch.full((bs,), float("inf")).to(self.id_generator.device),
            query=1 - posdoc_sampling_target,
        )
