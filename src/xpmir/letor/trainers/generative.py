from typing import Iterator
from torch import nn
import numpy as np
from experimaestro import Param, Config
import torch

from xpmir.letor.samplers import PairwiseSampler
from xpmir.letor.records import BaseRecords, PairwiseRecords
from xpmir.neural.generative import IdentifierGenerator, StepwiseGenerator
from xpmir.letor.trainers import TrainerContext, LossTrainer
from xpmir.learning.context import Loss
from xpmir.utils.utils import foreach


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
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))


class PairwiseGenerativeRetrievalLoss(PairwiseGenerativeLoss):

    NAME = "PairwiseGenerativeLoss"

    id_generator: Param[IdentifierGenerator]

    def recursive(
        self,
        log_cur_node_proba,
        posdoc_stepwise_generator: StepwiseGenerator,
        negdoc_stepwise_generator: StepwiseGenerator,
        query_stepwise_generator: StepwiseGenerator,
    ):
        # pass get the probas
        log_posdoc_proba = posdoc_stepwise_generator.step()
        log_negdoc_proba = negdoc_stepwise_generator.step()
        log_query_proba = query_stepwise_generator.step()

        # middle_term in the formula
        middle_term = torch.sum(
            torch.exp(log_posdoc_proba + log_query_proba).detach()
            * (1 - torch.exp(log_negdoc_proba)).detach()
            * (
                torch.sum(log_cur_node_proba, dim=-1).unsqueeze(-1)
                + log_posdoc_proba
                + log_query_proba
            ),
            dim=-1,
        )  # shape: [bs, ]

        # last term in the formula
        sum_except_current = (
            torch.sum(
                torch.exp(log_posdoc_proba + log_query_proba).detach(), dim=-1
            ).unsqueeze(-1)
            - torch.exp(log_posdoc_proba + log_query_proba).detach()
        )
        last_term = torch.sum(
            torch.exp(log_negdoc_proba).detach()
            * sum_except_current
            * log_negdoc_proba,
            dim=-1,
        )  # shape: [bs, ]

        # TODO: get rid of this
        # obtain the previous unfinished sequence as a mask
        # 0 means no need to continue
        unfinished_sequences = posdoc_stepwise_generator.get_token_state()[1]

        # randomly choose the target of sampling
        sampling_target = int(torch.randint(low=0, high=3, size=(1,)))
        if sampling_target == 0:
            raw_next_tokens = torch.multinomial(
                torch.exp(log_posdoc_proba), num_samples=1
            )
        elif sampling_target == 1:
            raw_next_tokens = torch.multinomial(
                torch.exp(log_negdoc_proba), num_samples=1
            )
        elif sampling_target == 2:
            raw_next_tokens = torch.multinomial(
                torch.exp(log_query_proba), num_samples=1
            )

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

        # TODO: get rid of this: update the tokens for the next recursion
        # mask the generated tokens if some of the seqs
        # are already end before(0 in unfinished_sequences)
        posdoc_stepwise_generator.set_token_state(raw_next_tokens)
        negdoc_stepwise_generator.set_token_state(raw_next_tokens)
        query_stepwise_generator.set_token_state(raw_next_tokens)

        # whether need to be end now?
        if posdoc_stepwise_generator.stopping_criteria():
            return (middle_term + last_term) * unfinished_sequences.detach()

        if sampling_target == 0:
            sampling_multiplier = torch.exp(
                log_negdoc_proba_next_tokens + log_query_proba_next_tokens
            ).detach()
        elif sampling_target == 1:
            sampling_multiplier = torch.exp(
                log_posdoc_proba_next_tokens + log_query_proba_next_tokens
            ).detach()
        elif sampling_target == 2:
            sampling_multiplier = torch.exp(
                log_posdoc_proba_next_tokens + log_negdoc_proba_next_tokens
            ).detach()

        return unfinished_sequences.detach() * (
            self.recursive(
                log_cur_node_proba,
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

        # in fact, we need to minus something to get the pure gradient, but at
        # the level of the root, the additional terms always equals to 0
        return torch.mean(
            self.recursive(
                log_cur_node_proba,
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

# if __name__ == '__main__':
#     model = T5IdentifierGenerator(hf_id='t5-base')
#     model.add_pretasks(LoadFromT5(model=model))

#     loss = PairwiseGenerativeRetrievalLoss(id_generator=model)
#     loss = loss.instance()

#     input = PairwiseRecords()
#     p1 = PairwiseRecord(
#         TopicRecord(TextTopic("query")),
#         DocumentRecord(TextDocument("positive document")),
#         DocumentRecord(TextDocument("negative document"))
#     )
#     p2 = PairwiseRecord(
#         TopicRecord(TextTopic("this is one another")),
#         DocumentRecord(TextDocument("bug please go away")),
#         DocumentRecord(TextDocument("I don't like bugs"))
#     )

#     input.add(p1)
#     input.add(p2)

#     print(loss.compute(input, None))
