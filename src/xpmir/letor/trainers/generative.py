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
        value = self.compute(records, context)
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))


class PairwiseGenerativeRetrievalLoss(PairwiseGenerativeLoss):

    NAME = "PairwiseGenerativeLoss"

    id_generator: Param[IdentifierGenerator]

    def recursive(
        self,
        cur_node_proba,
        posdoc_stepwise_generator: StepwiseGenerator,
        negdoc_stepwise_generator: StepwiseGenerator,
        query_stepwise_generator: StepwiseGenerator,
    ):
        # pass get the probas
        posdoc_proba = posdoc_stepwise_generator.step()
        negdoc_proba = negdoc_stepwise_generator.step()
        query_proba = query_stepwise_generator.step()

        # middle_term in the formula
        middle_term = torch.sum(
            posdoc_proba.detach()
            * query_proba.detach()
            * (1 - negdoc_proba).detach()
            * (
                torch.prod(torch.log(cur_node_proba), dim=-1).unsqueeze(-1)
                + torch.log(posdoc_proba * query_proba)
            ),
            dim=-1,
        )  # shape: [bs, tree_width]

        # last term in the formula
        sum_except_current = (
            torch.sum(posdoc_proba.detach() * query_proba.detach(), dim=-1).unsqueeze(
                -1
            )
            - posdoc_proba.detach() * query_proba.detach()
        )
        last_term = torch.sum(
            negdoc_proba.detach() * sum_except_current * torch.log(negdoc_proba), dim=-1
        )  # shape: [bs, tree_width]

        # obtain the previous unfinished sequence as a mask
        # 0 means no need to continue
        unfinished_sequences = posdoc_stepwise_generator.get_token_state()[1]

        # randomly choose the target of sampling
        sampling_target = torch.randint(low=0, high=3, size=(1,))
        raw_next_tokens = torch.cat(
            (
                torch.multinomial(posdoc_proba, num_samples=1),
                torch.multinomial(negdoc_proba, num_samples=1),
                torch.multinomial(query_proba, num_samples=1),
            ),
            dim=-1,
        )[:, sampling_target].squeeze(
            1
        )  # shape [bs]

        # mask the generated tokens if some of the seqs
        # are already end before(0 in unfinished_sequences)
        posdoc_stepwise_generator.set_token_state(raw_next_tokens)
        negdoc_stepwise_generator.set_token_state(raw_next_tokens)
        query_stepwise_generator.set_token_state(raw_next_tokens)

        # get the processed tokens
        next_tokens = posdoc_stepwise_generator.get_token_state()[0].squeeze(-1)

        # cumulate the proba from root
        iterator_vector = torch.arange(len(next_tokens))
        posdoc_proba_next_tokens = posdoc_proba[iterator_vector, next_tokens]
        negdoc_proba_next_tokens = negdoc_proba[iterator_vector, next_tokens]
        query_proba_next_tokens = query_proba[iterator_vector, next_tokens]
        cur_node_proba = cur_node_proba * torch.vstack(
            (
                posdoc_proba_next_tokens,
                negdoc_proba_next_tokens,
                query_proba_next_tokens,
            )
        ).transpose(0, 1)

        # whether need to be end now?
        if posdoc_stepwise_generator.stopping_criteria():
            return (middle_term + last_term) * unfinished_sequences.detach()

        sampling_multiplier = torch.vstack(
            (
                negdoc_proba_next_tokens.detach() * query_proba_next_tokens.detach(),
                posdoc_proba_next_tokens.detach() * query_proba_next_tokens.detach(),
                posdoc_proba_next_tokens.detach() * negdoc_proba_next_tokens.detach(),
            )
        )[sampling_target].squeeze(0)

        return unfinished_sequences.detach() * (
            self.recursive(
                cur_node_proba,
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
        cur_node_proba = torch.ones((bs, 3), dtype=torch.long).to(
            self.id_generator.device
        )

        # in fact, we need to minus something to get the pure gradient, but at
        # the level of the root, the additional terms always equals to 0
        return self.recursive(
            cur_node_proba,
            posdoc_stepwise_generator,
            negdoc_stepwise_generator,
            query_stepwise_generator,
        )


class GenerativeTrainer(LossTrainer):

    loss: Param[PairwiseGenerativeRetrievalLoss]

    sampler: Param[PairwiseSampler]
    """The pairwise sampler"""

    def initialize(self, random: np.random.RandomState, context: TrainerContext):
        super().initialize(random, context)
        self.lossfn.initialize()
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

    def train_iter(self, records: PairwiseRecords):
        # do the forward pass to get the gradient value
        self.loss.process(records, self.context)
