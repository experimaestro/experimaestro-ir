from typing import List
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

    max_depth: Param[int] = 5

    def initialize(self):
        pass

    def recursive(
        self,
        attention_mask: List,  # the input mask of the pos, neg, query
        encoder_outputs: List,  # the encoder input of the pos, neg, query
        decoder_input_ids,  # tensor of shape [bs, 1] or None,
        past_key_values: List,  # the past_key_values or None
        cur_node_proba,  # tensor of shape [bs, 3]
        sampling_target,  # sampling from the distribution of which input,
        unfinished_sequences,  # tensor of shape [bs]
        depth,
    ):
        # pass get the probas and corresponding kvs
        posdoc_proba, posdoc_pkv = self.id_generator(
            attention_mask[0],
            encoder_outputs[0],
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values[0],
        )
        negdoc_proba, negdoc_pkv = self.id_generator(
            attention_mask[1],
            encoder_outputs[1],
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values[1],
        )
        query_proba, query_pkv = self.id_generator(
            attention_mask[2],
            encoder_outputs[2],
            decoder_input_ids=decoder_input_ids,
            past_key_values=past_key_values[2],
        )

        # gather the pkv
        past_key_values = [posdoc_pkv, negdoc_pkv, query_pkv]

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

        # sampling according to the proba distribution
        if sampling_target == "pos":
            next_tokens = torch.multinomial(posdoc_proba, num_samples=1).squeeze(
                1
            )  # shape bs
        elif sampling_target == "neg":
            next_tokens = torch.multinomial(negdoc_proba, num_samples=1).squeeze(
                1
            )  # shape bs
        elif sampling_target == "query":
            next_tokens = torch.multinomial(query_proba, num_samples=1).squeeze(
                1
            )  # shape bs
        else:
            raise ValueError(f"We cannot sampling over {sampling_target}")

        # mask the generated tokens if some of the seqs is already end before
        next_tokens = (
            next_tokens * unfinished_sequences
            + self.id_generator.pad_token_id * (1 - unfinished_sequences)
        )

        # test if it reaches the eos, then
        # solution: applying the mask
        # end the recursive!!!
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

        # check whether some seqs reach the eos at the end of this turn
        new_unfinished_sequences = unfinished_sequences.mul(
            next_tokens.tile(1, 1)
            .ne(torch.tensor([[self.id_generator.eos_token_id]]))
            .prod(dim=0)
        )

        # end the recursive if all the posdoc is finish or reaches the max_length
        if new_unfinished_sequences.max() == 0 or depth == self.max_depth:
            return (middle_term + last_term) * unfinished_sequences

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
            unfinished_sequences.detach()
            * (  # Question: does the mask need to detach()?
                self.recursive(
                    attention_mask,
                    encoder_outputs,
                    next_tokens.unsqueeze(-1),
                    past_key_values,
                    cur_node_proba,
                    sampling_target,
                    new_unfinished_sequences,
                    depth + 1,
                )
                * sample_multiplier
                + middle_term
                + last_term
            )
        )

    def compute(self, records: PairwiseRecords, context: TrainerContext):

        posdocs_text = [pdr.document.get_text() for pdr in records.positives]
        negdocs_text = [ndr.document.get_text() for ndr in records.negatives]
        queries_text = [qr.topic.get_text() for qr in records.unique_queries]

        bs = len(posdocs_text)

        pos_encoder_output, pos_attention_mask = self.id_generator.encode(posdocs_text)
        neg_encoder_output, neg_attention_mask = self.id_generator.encode(negdocs_text)
        qry_encoder_output, qry_attention_mask = self.id_generator.encode(queries_text)

        encoder_outputs = [pos_encoder_output, neg_encoder_output, qry_encoder_output]
        attention_mask = [pos_attention_mask, neg_attention_mask, qry_attention_mask]

        cur_node_proba = torch.ones((bs, 3), dtype=torch.long)
        past_key_values = [None, None, None]
        decoder_input_ids = None

        # vector of len bs filled with 0 or 1,
        # 1 represent not finish yet and 0 represent already finish
        unfinished_sequences = torch.ones(bs, dtype=torch.long)

        # in fact, we need to minus something to get the pure gradient, but at
        # the level of the root, the additional terms always equals to 0
        return self.recursive(
            attention_mask,
            encoder_outputs,
            decoder_input_ids,
            past_key_values,
            cur_node_proba,
            "pos",
            unfinished_sequences,
            0,
        )

    def process(self, records: PairwiseRecords, context: TrainerContext):
        """Calculate the loss and put it on the training context"""
        value = self.compute(records, context)
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))
