from typing import Iterator
from torch import nn
import numpy as np
from experimaestro import Param, Config
import torch
import logging

from xpmir.letor.samplers import PairwiseSampler
from xpmir.letor.records import BaseRecords, PairwiseRecords
from xpmir.neural.generative import IdentifierGenerator, StepwiseGenerator
from xpmir.letor.trainers import TrainerContext, LossTrainer
from xpmir.learning.context import Loss
from xpmir.utils.utils import foreach, easylog

logger = easylog()


class PairwiseGenerativeLoss(Config, nn.Module):
    """Generic loss for generative models"""

    NAME = "?"

    weight: Param[float] = 1.0
    """The weight :math:`w` with which the loss is multiplied (useful when
    combining with other ones)"""

    def initialize(self):
        pass

    def compute(self, records, context):
        pass

    def process(self, records: BaseRecords, context: TrainerContext):
        value = self.compute(records, context)  # tensor shape [bs]
        logger.debug(f"Loss: {value}")
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))


class PairwiseGenerativeRetrievalLoss(PairwiseGenerativeLoss):

    NAME = "PairwiseGenerativeLoss"

    id_generator: Param[IdentifierGenerator]
    """The id generator"""

    max_depth: Param[int] = 5
    """The max number of the steps we need to consider"""

    def recursive(
        self,
        decoder_input_tokens,  # None or [bs]
        unfinished_sequences: torch.tensor,  # shape [bs]
        log_cur_node_proba: torch.tensor,  # shape [bs,3]
        depth: int,
        posdoc_stepwise_generator: StepwiseGenerator,
        negdoc_stepwise_generator: StepwiseGenerator,
        query_stepwise_generator: StepwiseGenerator,
    ):
        # pass get the probas
        log_posdoc_proba = posdoc_stepwise_generator.step(decoder_input_tokens)
        log_negdoc_proba = negdoc_stepwise_generator.step(decoder_input_tokens)
        log_query_proba = query_stepwise_generator.step(decoder_input_tokens)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("\n")
            logger.debug(f"posdoc_proba: {torch.exp(log_posdoc_proba)}")
            logger.debug(f"negdoc_proba: {torch.exp(log_negdoc_proba)}")
            logger.debug(f"query_proba: {torch.exp(log_query_proba)}")

        # middle_term in the formula
        middle_term = torch.sum(
            torch.exp(log_posdoc_proba.detach() + log_query_proba.detach())
            * (1 - torch.exp(log_negdoc_proba.detach()))
            * (
                torch.sum(log_cur_node_proba, dim=-1).unsqueeze(-1)
                + log_posdoc_proba
                + log_query_proba
            ),
            dim=-1,
        )  # shape: [bs]

        # last term in the formula
        max_values_tmp = torch.max(
            log_posdoc_proba.detach() + log_query_proba.detach(), dim=-1
        ).values
        sum_except_current = (
            torch.sum(
                torch.exp(
                    log_posdoc_proba.detach()
                    + log_query_proba.detach()
                    - max_values_tmp.unsqueeze(-1)
                ),
                dim=-1,
            )
            * torch.exp(max_values_tmp)
        ).unsqueeze(-1) - torch.exp(
            log_posdoc_proba.detach() + log_query_proba.detach()
        )
        last_term = torch.sum(
            torch.exp(log_negdoc_proba.detach())
            * sum_except_current
            * log_negdoc_proba,
            dim=-1,
        )  # shape: [bs]

        # randomly choose the target of sampling
        sampling_target = int(torch.randint(low=0, high=3, size=(1,)))
        if sampling_target == 0:
            raw_next_tokens = torch.multinomial(
                torch.exp(log_posdoc_proba), num_samples=1
            ).squeeze(1)
        elif sampling_target == 1:
            raw_next_tokens = torch.multinomial(
                torch.exp(log_negdoc_proba), num_samples=1
            ).squeeze(1)
        elif sampling_target == 2:
            raw_next_tokens = torch.multinomial(
                torch.exp(log_query_proba), num_samples=1
            ).squeeze(1)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"sampled token is {raw_next_tokens} of depth {depth}")

        # Here we need to use the raw token to calculate
        # to avoid the index out of bound pb (it will be masked anyways)
        # cumulate the proba from root
        iterator_vector = torch.arange(len(raw_next_tokens))
        log_posdoc_proba_next_tokens = log_posdoc_proba[
            iterator_vector, raw_next_tokens
        ]
        log_negdoc_proba_next_tokens = log_negdoc_proba[
            iterator_vector, raw_next_tokens
        ]
        log_query_proba_next_tokens = log_query_proba[iterator_vector, raw_next_tokens]
        log_cur_node_proba = log_cur_node_proba + torch.vstack(
            (
                log_posdoc_proba_next_tokens,
                log_negdoc_proba_next_tokens,
                log_query_proba_next_tokens,
            )
        ).transpose(0, 1)

        # mask the generated tokens if some of the seqs
        # are already end before(0 in unfinished_sequences)
        raw_next_tokens = (
            raw_next_tokens * unfinished_sequences
            + self.id_generator.pad_token_id * (1 - unfinished_sequences)
        )
        new_unfinished_sequences = unfinished_sequences.mul(
            raw_next_tokens.tile(1, 1)
            .ne(
                torch.tensor([self.id_generator.eos_token_id]).to(
                    self.id_generator.device
                )
            )
            .prod(dim=0)
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"input token for the next step: {raw_next_tokens}")

        # whether need to be end now?
        if new_unfinished_sequences.max() == 0 or depth == self.max_depth:
            return (middle_term + last_term) * unfinished_sequences.detach()

        if sampling_target == 0:
            sampling_multiplier = torch.exp(
                log_negdoc_proba_next_tokens.detach()
                + log_query_proba_next_tokens.detach()
            )
        elif sampling_target == 1:
            sampling_multiplier = torch.exp(
                log_posdoc_proba_next_tokens.detach()
                + log_query_proba_next_tokens.detach()
            )
        elif sampling_target == 2:
            sampling_multiplier = torch.exp(
                log_posdoc_proba_next_tokens.detach()
                + log_negdoc_proba_next_tokens.detach()
            )

        return unfinished_sequences.detach() * (
            self.recursive(
                raw_next_tokens,
                new_unfinished_sequences,
                log_cur_node_proba,
                depth + 1,
                posdoc_stepwise_generator,
                negdoc_stepwise_generator,
                query_stepwise_generator,
            )
            * sampling_multiplier
            + middle_term
            + last_term
        )

    def compute(self, records: PairwiseRecords, context: TrainerContext):

        posdocs_text = [pdr.document.get_text() for pdr in records.positives]
        negdocs_text = [ndr.document.get_text() for ndr in records.negatives]
        queries_text = [qr.topic.get_text() for qr in records.unique_queries]

        bs = len(posdocs_text)

        logger.debug(f"posdocs_text: {posdocs_text}")
        logger.debug(f"negdocs_text: {negdocs_text}")
        logger.debug(f"queries_text: {queries_text}")

        # create the generator for the given records
        posdoc_stepwise_generator = self.id_generator.stepwise_iterator()
        negdoc_stepwise_generator = self.id_generator.stepwise_iterator()
        query_stepwise_generator = self.id_generator.stepwise_iterator()

        posdoc_stepwise_generator.init(posdocs_text)
        negdoc_stepwise_generator.init(negdocs_text)
        query_stepwise_generator.init(queries_text)

        # initialize cumulate product of from the root to the current one
        log_cur_node_proba = torch.zeros((bs, 3), dtype=torch.long).to(
            self.id_generator.device
        )

        # initialize the input for the decoder and the mask
        decoder_input_tokens = None
        unfinished_sequences = torch.ones(bs, dtype=torch.long).to(
            self.id_generator.device
        )

        # in fact, we need to minus something to get the pure gradient, but at
        # the level of the root, the additional terms always equals to 0
        return -torch.mean(
            self.recursive(
                decoder_input_tokens,
                unfinished_sequences,
                log_cur_node_proba,
                1,
                posdoc_stepwise_generator,
                negdoc_stepwise_generator,
                query_stepwise_generator,
            )
        )


class GenerativeTrainer(LossTrainer):

    loss: Param[PairwiseGenerativeRetrievalLoss]

    sampler: Param[PairwiseSampler]
    """The pairwise sampler"""

    def initialize(self, random: np.random.RandomState, context: TrainerContext):
        super().initialize(random, context)
        self.loss.initialize()
        foreach(
            context.hooks(PairwiseGenerativeLoss), lambda loss: loss.initialize()
        )  # maybe later we need to change the sampling target, we can use this hook

        self.sampler.initialize(random)
        self.sampler_iter = self.sampler.pairwise_iter()

    def iter_batches(self) -> Iterator[PairwiseRecords]:
        while True:
            batch = PairwiseRecords()
            for _, record in zip(range(self.batch_size), self.sampler_iter):
                batch.add(record)
            yield batch

    def train_batch(self, records: PairwiseRecords):
        # do the forward pass to get the gradient value
        self.loss.process(records, self.context)


# # to test
# from xpmir.neural.generative.hf import LoadFromT5, T5IdentifierGenerator
# from datamaestro_text.data.ir.base import TextDocument, TextTopic
# from xpmir.letor.records import PairwiseRecord, TopicRecord, DocumentRecord

# if __name__ == "__main__":
#     model = T5IdentifierGenerator(hf_id="t5-base")
#     model.add_pretasks(LoadFromT5(model=model))

#     loss = PairwiseGenerativeRetrievalLoss(id_generator=model, max_depth=5)
#     loss = loss.instance()

#     input = PairwiseRecords()
#     p1 = PairwiseRecord(
#         TopicRecord(TextTopic("query")),
#         DocumentRecord(TextDocument("positive document")),
#         DocumentRecord(TextDocument("negative document")),
#     )
#     p2 = PairwiseRecord(
#         TopicRecord(TextTopic("this is one another")),
#         DocumentRecord(TextDocument("bug please go away")),
#         DocumentRecord(TextDocument("I don't like bugs")),
#     )

#     input.add(p1)
#     input.add(p2)

#     print(loss.compute(input, None))
