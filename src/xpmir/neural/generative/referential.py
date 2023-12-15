import logging
from collections import namedtuple
import dataclasses
from dataclasses import dataclass
from typing import Generic, List, NamedTuple, TypeVar, Optional

import torch
from experimaestro import Config, Param
from experimaestro.compat import cached_property
from torch import nn, LongTensor, FloatTensor
from transformers import LogitsProcessor, LogitsProcessorList

from xpmir.learning.context import Loss, StepTrainingHook
from xpmir.learning.metrics import ScalarMetric
from xpmir.letor.records import BaseRecords, PairwiseRecords
from xpmir.letor.trainers import TrainerContext
from xpmir.letor.trainers.generative import PairwiseGenerativeLoss
from xpmir.neural.generative import (
    ConditionalGenerator,
    GenerateOptions,
    StepwiseGenerator,
    BeamSearchGenerationOptions,
)
from xpmir.neural.generative.hf import FullSequenceGenerationOutput
from xpmir.learning import ModuleInitOptions
from xpmir.rankers import AbstractModuleScorer
from xpmir.utils.utils import easylog
from xpmir.documents.samplers import UpdatableRandomSpanSampler
from xpmir.context import Hook

logger = easylog()

T = TypeVar("T")


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

    def __post_init__(self):
        # if no update
        self.current_max_depth = self.max_depth


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


# dataclass for inference, compose of doc and qry
PairwiseTuple = namedtuple("PairwiseTuple", ["doc", "qry"])


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


class DepthBasedSequenceBiasLogitsProcessor(LogitsProcessor):
    """Only support the bias term of length 1, and the bias is only added at the eos"""

    def __init__(self, sequence_bias: torch.tensor) -> None:
        super().__init__()
        self.sequence_bias = sequence_bias  # shape [bs*num_beam, decoder_dim+1]

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        # We don't need to renormalize the logits here, we can just set
        # renormalize_logits=True during the generation
        current_depth = input_ids.shape[1]  # counting from 1, 2, ..
        return scores + self.sequence_bias[current_depth - 1]


# The stepwise generator for the model with additional bias
class GeneratorBiasStepwiseGenerator(StepwiseGenerator):
    def __init__(
        self,
        id_generator: ConditionalGenerator,
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
class GeneratorBiasAdapter(ConditionalGenerator):
    max_depth: Param[int] = 5
    """The max_depth of the generator"""

    vanilla_generator: Param[ConditionalGenerator]
    """The original generator, for the moment a T5ConditionalGenerator"""

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.vanilla_generator.initialize(options)
        self.decoder_outdim = self.vanilla_generator.decoder_outdim
        self.eos_token_id = self.vanilla_generator.eos_token_id
        self.pad_token_id = self.vanilla_generator.pad_token_id
        self.decoder_start_token_id = self.vanilla_generator.decoder_start_token_id

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
                sum(decoder_dim**i for i in range(j))
                for j in range(self.max_depth, 0, -1)
            ]
        ).to(self.device)
        alphas = torch.log((1 / alphas)).unsqueeze(-1)
        return torch.cat(
            (torch.zeros(alphas.shape[0], decoder_dim).to(self.device), alphas), -1
        )

    def generate(self, inputs: List[str], options: GenerateOptions = None):
        assert options.return_dict_in_generate, "Must return dict in this case"
        assert options.output_scores, "Must return scores in this case"

        # prepare the LogitsProcessor
        logit_processor_list = LogitsProcessorList(
            [DepthBasedSequenceBiasLogitsProcessor(self.bias_terms)]
        )
        inputs = self.vanilla_generator.batch_tokenize(inputs, mask=True)
        generate_options_kwargs = dataclasses.asdict(options)
        if isinstance(options, BeamSearchGenerationOptions):
            res = self.vanilla_generator.model.generate(
                input_ids=inputs.ids,
                attention_mask=inputs.mask,
                renormalize_logits=True,  # important,
                logits_processor=logit_processor_list,
                **generate_options_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Generation Options not supported for {options.__class__}"
            )

        # -- For the old version should be compute_transition_beam_scores
        output_mask = torch.where(res.sequences != self.pad_token_id, 1, 0).to(
            self.device
        )
        if self.pad_token_id == self.decoder_start_token_id:
            output_mask[:, 0] = 1
        transition_scores = self.vanilla_generator.model.compute_transition_scores(
            res.sequences,
            res.scores,
            res.beam_indices,
            normalize_logits=False,  # for bs the logits are already normalized
        )
        full_score = torch.sum(transition_scores, dim=-1)
        return FullSequenceGenerationOutput(
            sequences=res.sequences,
            output_mask=output_mask,
            transition_scores=transition_scores,
            all_scores=res.scores,
            sequence_scores=full_score,
        )


# -- Hooks
class LossProcessHook(Hook):
    def process(self):
        """Called after process"""
        pass


class NegativeUpdateHook(LossProcessHook):

    sampler: Param[UpdatableRandomSpanSampler]
    """The sampler which contains the hard negatives"""

    max_depth: Param[int]
    """The max depth for the model"""

    def process(
        self,
        loss_output: GenerativeLossOutput,
        ids: list[int],
        current_max_depth: int,
    ):
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
                torch.tensor(ids, dtype=torch.int32),  # shape [bs]
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

    def recursive(
        self,
        decoder_input_tokens,  # None or [depth - 1, bs]
        unfinished_sequences: torch.tensor,  # shape [bs]
        log_cur_node_proba: Triplet[torch.Tensor],  # shape [bs,3]
        depth: int,  # a value counting from 1
        stepwise_generators: Triplet[StepwiseGenerator],
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
        ids = [pdr.document.get_id() for pdr in records.positives]

        for hook in self.loss_hooks:
            hook.process(loss_output, ids, self.current_max_depth)


class GenerativeRetrievalScorer(AbstractModuleScorer, DepthUpdatable):
    """The abstract class for the generative retrieval scorer"""

    id_generator: Param[ConditionalGenerator]
    """The id generator"""

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.id_generator.initialize(options)

    def update_depth(self, new_depth):
        if new_depth <= self.max_depth:
            self.current_max_depth = new_depth
            logger.info(
                f"Update the max_depth to {self.current_max_depth} for the scorer"
            )
        else:
            self.current_max_depth = self.max_depth


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
        # -- the exact term of the scorer
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


class BeamSearchRandomBasedGenerativeRetrievalScorer(
    RandomBasedGenerativeRetrievalScorer
):
    """A scorer based on the probability that a document is better than a random
    document given a query, but we choose plusieur paths for the query"""

    num_beams: Param[int] = 16

    def recursive(
        self,
        depth,  # a value counting from 1
        stepwise_generator: StepwiseGenerator,
        query_generate_output: FullSequenceGenerationOutput,
    ):
        assert depth <= self.current_max_depth
        decoder_input_tokens = (
            None if depth == 1 else query_generate_output.sequences[:, depth - 1]
        )
        # pass get the probas
        document_logits = stepwise_generator.step(decoder_input_tokens)
        document_log_proba = nn.functional.log_softmax(
            document_logits, dim=-1
        )  # [bs*num_beam, vocab_size]
        log_proba_randdoc = self.random_distribution[depth - 1]

        # print(document_log_proba.shape)

        # -- the exact term of the scorer
        exact_term = torch.sum(
            torch.exp(query_generate_output.all_scores[depth - 1] + document_log_proba)
            * (1 - torch.exp(log_proba_randdoc)),
            dim=-1,
        )

        # get the tokens passed for the next layers --> predefined by query
        next_tokens = query_generate_output.sequences[:, depth]  # shape [bs*num_beam]
        # print(next_tokens.shape)

        batch_range = torch.arange(len(next_tokens))

        # print(batch_range)
        log_proba_next_doc = document_log_proba[batch_range, next_tokens]

        log_proba_next_randdoc = torch.where(
            next_tokens == self.id_generator.eos_token_id,
            log_proba_randdoc[self.id_generator.eos_token_id],
            log_proba_randdoc[0],
        )
        unfinished_sequences = query_generate_output.output_mask[:, depth - 1]
        new_unfinished_sequences = query_generate_output.output_mask[:, depth]
        if new_unfinished_sequences.max() == 0 or depth == self.current_max_depth:
            return unfinished_sequences * exact_term

        return unfinished_sequences * (
            self.recursive(
                depth + 1,
                stepwise_generator,
                query_generate_output,
            )
            * torch.exp(log_proba_next_doc + log_proba_next_randdoc)
            + exact_term
        )

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        self.id_generator.eval()
        queries_text = [pdr.topic.get_text() for pdr in inputs.topics]
        query_generate_options = BeamSearchGenerationOptions(
            max_new_tokens=self.current_max_depth,
            num_return_sequences=self.num_beams,
            num_beams=self.num_beams,
        )
        query_generate_output: FullSequenceGenerationOutput = (
            self.id_generator.generate(queries_text, query_generate_options)
        )
        documents_text = [ndr.document.get_text() for ndr in inputs.documents]
        documents_text_repeat = [
            d for d in documents_text for _ in range(self.num_beams)
        ]

        document_stepwise_generator = self.id_generator.stepwise_iterator()
        document_stepwise_generator.init(documents_text_repeat)

        recursive_output = self.recursive(
            1, document_stepwise_generator, query_generate_output
        )  # shape [bs, num_beams]

        # sequence_scores_sum = torch.sum(
        #     torch.exp(query_generate_output.sequence_scores).reshape(
        #         -1, self.num_beams
        #     ),
        #     dim=-1,
        # )
        # info.add_metric(
        #     ScalarMetric(
        #         "Query total portion", float(sequence_scores_sum), len(queries_text)
        #     )
        # )
        return torch.sum(recursive_output.reshape(-1, self.num_beams), -1)


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
