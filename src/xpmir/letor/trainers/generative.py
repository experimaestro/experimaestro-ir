from experimaestro import Param
import torch

from xpmir.letor.trainers.pairwise import PairwiseLoss
from xpmir.letor.records import PairwiseRecords
from xpmir.neural.generative import IdentifierGenerator
from xpmir.letor.trainers import TrainerContext
from xpmir.learning.context import Loss


class PairwiseGenerativeRetrievalLoss(PairwiseLoss):

    NAME = "PairwiseGenerativeLoss"

    id_generator: Param[IdentifierGenerator]

    def initialize(self):  # Question: what name of initialize method to use?
        self.posdoc_stepwise_generator = self.id_generator.stepwise_iterator()
        self.negdoc_stepwise_generator = self.id_generator.stepwise_iterator()
        self.query_stepwise_generator = self.id_generator.stepwise_iterator()

    def recursive(self, cur_node_proba, sampling_target):

        # pass get the probas
        posdoc_proba = self.posdoc_stepwise_generator.step()
        negdoc_proba = self.negdoc_stepwise_generator.step()
        query_proba = self.query_stepwise_generator.step()

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
        unfinished_sequences = self.posdoc_stepwise_generator.get_token_state()[1]

        # sampling according to the proba distribution
        if sampling_target == "pos":
            raw_next_tokens = torch.multinomial(posdoc_proba, num_samples=1).squeeze(
                1
            )  # shape bs
        elif sampling_target == "neg":
            raw_next_tokens = torch.multinomial(negdoc_proba, num_samples=1).squeeze(
                1
            )  # shape bs
        elif sampling_target == "query":
            raw_next_tokens = torch.multinomial(query_proba, num_samples=1).squeeze(
                1
            )  # shape bs
        else:
            raise ValueError(f"We cannot sampling over {sampling_target}")

        # mask the generated tokens if some of the seqs
        # are already end before(0 in unfinished_sequences)
        self.posdoc_stepwise_generator.set_token_state(raw_next_tokens)
        self.negdoc_stepwise_generator.set_token_state(raw_next_tokens)
        self.query_stepwise_generator.set_token_state(raw_next_tokens)

        # get the processed tokens
        next_tokens = self.posdoc_stepwise_generator.get_token_state()[0].squeeze(-1)

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
        if self.posdoc_stepwise_generator.stopping_criteria():
            # Question: does the mask need to detach()?
            return (middle_term + last_term) * unfinished_sequences.detach()

        if sampling_target == "pos":
            sample_multiplier = (
                negdoc_proba_next_tokens.detach() * query_proba_next_tokens.detach()
            )
        elif sampling_target == "neg":
            sample_multiplier = (
                posdoc_proba_next_tokens.detach() * query_proba_next_tokens.detach()
            )
        elif sampling_target == "query":
            sample_multiplier = (
                posdoc_proba_next_tokens.detach() * negdoc_proba_next_tokens.detach()
            )
        else:
            raise ValueError(f"We cannot sampling over {sampling_target}")

        return (
            # Question: does the mask need to detach()?
            unfinished_sequences.detach()
            * (
                self.recursive(cur_node_proba, sampling_target) * sample_multiplier
                + middle_term
                + last_term
            )
        )

    def compute(
        self, records: PairwiseRecords, context: TrainerContext, sampling_target: str
    ):

        posdocs_text = [pdr.document.get_text() for pdr in records.positives]
        negdocs_text = [ndr.document.get_text() for ndr in records.negatives]
        queries_text = [qr.topic.get_text() for qr in records.unique_queries]

        bs = len(posdocs_text)

        self.posdoc_stepwise_generator.init(posdocs_text)
        self.negdoc_stepwise_generator.init(negdocs_text)
        self.query_stepwise_generator.init(queries_text)

        # initialize cumulate product of from the root to the current one
        cur_node_proba = torch.ones((bs, 3), dtype=torch.long).to(
            self.id_generator.device
        )

        # in fact, we need to minus something to get the pure gradient, but at
        # the level of the root, the additional terms always equals to 0
        return self.recursive(cur_node_proba, sampling_target)

    def process(self, records: PairwiseRecords, context: TrainerContext):
        """Calculate the loss and put it on the training context"""
        value = self.compute(records, context)
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))
