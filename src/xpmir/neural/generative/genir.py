import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import Generic, List, NamedTuple, Optional, TypeVar

import numpy as np
import torch
from experimaestro import Config, Param
from experimaestro.compat import cached_property
from torch import nn

from xpmir.learning.context import Loss, StepTrainingHook
from xpmir.learning.metrics import ScalarMetric
from xpmir.letor.records import BaseRecords, PairwiseRecords
from xpmir.letor.trainers import TrainerContext
from xpmir.letor.trainers.generative import PairwiseGenerativeLoss
from xpmir.neural.generative import (
    GenerativeRetrievalScorer,
    IdentifierGenerator,
    StepwiseGenerator,
)
from xpmir.rankers import AbstractModuleScorer
from xpmir.utils.utils import easylog

logger = easylog()

T = TypeVar("T")

# dataclass for training, compose of pos_doc, neg_doc and query


class DepthUpdatable(Config):
    """Abstract class of the objects which could update their depth"""

    max_depth: Param[int] = 5
    """The max number of the steps we need to consider, counting from 1"""

    current_max_depth: int
    """The max_depth for the current learning stage in the progressive training
    stage"""

    def update_depth(self, new_depth):
        if new_depth <= self.max_depth:
            self.current_max_depth = new_depth
            logger.info(
                f"Update the max_depth to {self.current_max_depth} for the loss"
            )
        else:
            self.current_max_depth = self.max_depth

    def initialize(self):
        # if no update
        self.current_max_depth = self.max_depth


@dataclass
class Triplet(Generic[T]):
    pos_doc: T
    neg_doc: T
    query: T

    def __iter__(self):
        yield self.pos_doc
        yield self.neg_doc
        yield self.query


# dataclass for inference, compose of doc and qry
PairwiseTuple = namedtuple("PairwiseTuple", ["doc", "qry"])


class GenerativeLossOutput(NamedTuple):
    recursive_loss: torch.tensor
    kl_div_loss: torch.tensor
    pairwise_accuracy: torch.tensor


# The stepwise generator for the model with additional bias
class GeneratorBiasStepwiseGenerator(StepwiseGenerator):
    def __init__(
        self,
        id_generator: IdentifierGenerator,
        stepwise_iterator: StepwiseGenerator,
    ):
        super().__init__()
        # The identifier to use to generate the next step's token
        self.id_generator = id_generator
        self.stepwise_iterator = stepwise_iterator

    def init(self, texts: List[str]):
        self.stepwise_iterator.init(texts)
        self.current_depth = 0

    def step(self, token_ids: torch.LongTensor) -> torch.Tensor:
        logits = self.stepwise_iterator.step(token_ids)
        bs = logits.shape[0]
        logits = logits + self.id_generator.bias_terms[self.current_depth].expand(
            bs, -1
        )
        self.current_depth += 1
        return logits


# The model with addtional bias
class GeneratorBiasAdapter(IdentifierGenerator):
    max_depth: Param[int] = 5
    """The max_depth of the generator"""

    vanilla_generator: Param[IdentifierGenerator]
    """The original generator"""

    def __initialize__(self, random: Optional[np.random.RandomState] = None):
        super().__initialize__()
        self.vanilla_generator.initialize(random)
        self.decoder_outdim = self.vanilla_generator.decoder_outdim
        self.eos_token_id = self.vanilla_generator.eos_token_id
        self.pad_token_id = self.vanilla_generator.pad_token_id

    def stepwise_iterator(self) -> StepwiseGenerator:
        return GeneratorBiasStepwiseGenerator(
            self, self.vanilla_generator.stepwise_iterator()
        )

    @property
    def device(self):
        return self.vanilla_generator.device

    @cached_property
    def bias_terms(self):
        assert (
            self.vanilla_generator.eos_token_id == self.vanilla_generator.decoder_outdim
        )
        decoder_dim = self.vanilla_generator.decoder_outdim
        alphas = torch.tensor(
            [
                sum(decoder_dim**i for i in range(j + 1))
                for j in range(self.max_depth, 0, -1)
            ]
        ).to(self.device)
        alphas = torch.log((1 / alphas)).unsqueeze(-1)
        return torch.cat(
            (torch.zeros(alphas.shape[0], decoder_dim).to(self.device), alphas), -1
        )


# --- For training
class PairwiseGenerativeRetrievalLoss(PairwiseGenerativeLoss, DepthUpdatable):
    NAME = "PairwiseGenerativeLoss"

    id_generator: Param[IdentifierGenerator]
    """The id generator"""

    alpha: Param[float] = 0.0
    """The hyperparameter for the KL divergence"""

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
        super().initialize()
        super(DepthUpdatable).initialize()
        self.kl_lossfn = nn.KLDivLoss(reduction="sum", log_target=True)

    def recursive(
        self,
        decoder_input_tokens,  # None or [bs]
        unfinished_sequences: torch.tensor,  # shape [bs]
        log_cur_node_proba: Triplet[torch.Tensor],  # shape [bs,3]
        depth: int,  # a value counting from 1
        stepwise_generators: Triplet[StepwiseGenerator],
    ) -> GenerativeLossOutput:
        """Return the recursive G and the kl_div loss at depth, also the
        pairwise accruracy for supervision"""
        assert depth <= self.current_max_depth
        # pass get the probas, each one of shape: [bs, dec_dim+1]
        logits = Triplet(*(g.step(decoder_input_tokens) for g in stepwise_generators))

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
            log_p_next_token = torch.stack(  # shape [3, bs, dec_dim-1]
                (
                    log_proba.pos_doc[:, :-1],
                    # Avoids for now sampling from the negative distribution
                    # log_proba.neg_doc[:, :-1],
                    log_proba.query[:, :-1],
                )
            ).logsumexp(dim=0)

            next_tokens = torch.multinomial(
                torch.exp(log_p_next_token), num_samples=1
            ).squeeze(1)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"sampled token is {next_tokens} of depth {depth}")

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
            next_tokens,
            new_unfinished_sequences,
            log_cur_node_proba,
            depth + 1,
            stepwise_generators,
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

        # in fact, we need to minus something to get the pure gradient, but at
        # the level of the root, the additional terms always equals to 0
        loss = self.recursive(
            decoder_input_tokens,
            unfinished_sequences,
            log_cur_node_proba,
            1,
            stepwise_generators,
        )

        return GenerativeLossOutput(
            recursive_loss=-torch.mean(loss.recursive_loss),
            kl_div_loss=loss.kl_div_loss,
            pairwise_accuracy=torch.mean(loss.pairwise_accuracy),
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


# --- For inference
class NaiveGenerativeRetrievalScorer(GenerativeRetrievalScorer):
    """A naive scorer which will be used for the inference of the generative
    retrieval model, and this scorer is not learnable"""

    early_finish_punishment: Param[int] = 1
    """A early finish punishment hyperparameter, trying to make model score less
    if the id list is too short. Default value to 1 means no punishment"""

    def recursive(
        self,
        decoder_input_tokens,  # shape [bs]
        unfinished_sequences,
        depth,  # a value counting from 1
        stepwise_generators: PairwiseTuple,
    ):
        assert depth <= self.current_max_depth
        # pass get the probas
        logits = PairwiseTuple(
            *[g.step(decoder_input_tokens) for g in stepwise_generators]
        )
        log_proba = PairwiseTuple(
            *(nn.functional.log_softmax(logit, dim=-1) for logit in logits)
        )

        # take the maximum indices
        next_tokens = torch.max(torch.exp(log_proba.qry), dim=-1).indices

        batch_range = torch.arange(len(next_tokens))
        log_proba_next = PairwiseTuple(
            *(x[batch_range, next_tokens] for x in log_proba)
        )

        # mask them! For the sequence already finished, replaced by the
        # multiplier 1(or some other multiplier to punish the early finish)
        log_proba_next.doc[unfinished_sequences == 0] = self.early_finish_punishment
        log_proba_next.qry[unfinished_sequences == 0] = self.early_finish_punishment

        new_unfinished_sequences = (
            next_tokens != self.id_generator.eos_token_id
        ) & unfinished_sequences

        if new_unfinished_sequences.max() == 0 or depth == self.current_max_depth:
            return torch.exp(log_proba_next.qry)

        return self.recursive(
            next_tokens,
            new_unfinished_sequences,
            depth + 1,
            stepwise_generators,
        ) * torch.exp(log_proba_next.doc)

    def forward(
        self, inputs: "BaseRecords", info: TrainerContext = None
    ):  # try to return tensor [bs, ] which contains the scores
        # also implemented in a recursive way
        self.id_generator.eval()
        queries_text = [pdr.topic.get_text() for pdr in inputs.topics]
        documents_text = [ndr.document.get_text() for ndr in inputs.documents]

        bs = len(queries_text)

        stepwise_generator = PairwiseTuple(
            *[self.id_generator.stepwise_iterator() for _ in range(2)]
        )

        stepwise_generator.qry.init(queries_text)
        stepwise_generator.doc.init(documents_text)

        # initialization
        decoder_input_tokens = None
        unfinished_sequences = torch.ones(bs, dtype=torch.long).to(
            self.id_generator.device
        )

        return self.recursive(
            decoder_input_tokens, unfinished_sequences, 1, stepwise_generator
        )


# --- For inference
class RandomBasedGenerativeRetrievalScorer(GenerativeRetrievalScorer):
    """A scorer based on the probability that a document is better than a random
    document given a query, and this scorer is not learnable"""

    @cached_property
    def random_distribution(self):  # shape [max_depth, decoder_outdim+1]
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

    def recursive(
        self,
        decoder_input_tokens,  # shape [bs]
        unfinished_sequences,
        depth,  # a value counting from 1
        stepwise_generators: PairwiseTuple,
    ):
        assert depth <= self.current_max_depth
        # pass get the probas
        logits = PairwiseTuple(
            *[g.step(decoder_input_tokens) for g in stepwise_generators]
        )
        log_proba = PairwiseTuple(
            *(nn.functional.log_softmax(logit, dim=-1) for logit in logits)
        )

        log_proba_randdoc = self.random_distribution[depth - 1]
        # -- the exact term of the loss
        exact_term = torch.sum(
            torch.exp(log_proba.qry + log_proba.doc)
            * (1 - torch.exp(log_proba_randdoc)),
            dim=-1,
        )

        # take the sampled indices
        next_tokens = torch.multinomial(
            torch.exp(log_proba.qry), num_samples=1
        ).squeeze(1)

        batch_range = torch.arange(len(next_tokens))
        log_proba_next = PairwiseTuple(
            *(x[batch_range, next_tokens] for x in log_proba)
        )

        new_unfinished_sequences = (
            next_tokens != self.id_generator.eos_token_id
        ) & unfinished_sequences

        if new_unfinished_sequences.max() == 0 or depth == self.current_max_depth:
            return unfinished_sequences * exact_term

        # generate the proba for the random doc, according to whether the
        # sampled one is the eos
        log_proba_next_randdoc = torch.where(
            next_tokens == self.id_generator.eos_token_id,
            log_proba_randdoc[self.id_generator.eos_token_id],
            log_proba_randdoc[0],
        )

        return unfinished_sequences * (
            self.recursive(
                next_tokens,
                new_unfinished_sequences,
                depth + 1,
                stepwise_generators,
            )
            * torch.exp(log_proba_next.doc + log_proba_next_randdoc)
            + exact_term
        )

    def forward(
        self, inputs: "BaseRecords", info: TrainerContext = None
    ):  # try to return tensor [bs, ] which contains the scores
        # also implemented in a recursive way
        self.id_generator.eval()
        queries_text = [pdr.topic.get_text() for pdr in inputs.topics]
        documents_text = [ndr.document.get_text() for ndr in inputs.documents]

        bs = len(queries_text)

        stepwise_generator = PairwiseTuple(
            *[self.id_generator.stepwise_iterator() for _ in range(2)]
        )

        stepwise_generator.qry.init(queries_text)
        stepwise_generator.doc.init(documents_text)

        # initialization
        decoder_input_tokens = None
        unfinished_sequences = torch.ones(bs, dtype=torch.long).to(
            self.id_generator.device
        )

        return self.recursive(
            decoder_input_tokens, unfinished_sequences, 1, stepwise_generator
        )


class GenerativeRetrievalScorer(AbstractModuleScorer, DepthUpdatable):
    """The abstract class for the generative retrieval scorer"""

    id_generator: Param[IdentifierGenerator]
    """The id generator"""

    def _initialize(self, random):
        self.id_generator.initialize()
        super(DepthUpdatable).initialize()

    def update_depth(self, new_depth):
        if new_depth <= self.max_depth:
            self.current_max_depth = new_depth
            logger.info(
                f"Update the max_depth to {self.current_max_depth} for the scorer"
            )
        else:
            self.current_max_depth = self.max_depth


class GenRetDepthUpdateHook(StepTrainingHook):
    """Update the depth of the training instance(loss, scorer, etc) procedure"""

    objects: Param[List[DepthUpdatable]]
    """The objects to update the depth during the learning procedure"""

    update_interval: Param[int] = 200
    """The interval to update the learning depth"""

    def before(self, state: TrainerContext):
        # start with depth 1
        if state.steps % (self.update_interval * state.steps_per_epoch) == 1:
            current_depth = (state.epoch - 1) // self.update_interval + 1
            for object in self.objects:
                object.update_depth(current_depth)
