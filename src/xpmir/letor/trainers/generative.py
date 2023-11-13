from typing import Iterator
from collections import namedtuple
from torch import nn
import numpy as np
from experimaestro import Param, Config
import torch
import logging

from xpmir.letor.samplers import PairwiseSampler
from xpmir.letor.records import BaseRecords, PairwiseRecords
from xpmir.neural.generative import IdentifierGenerator
from xpmir.letor.trainers import TrainerContext, LossTrainer
from xpmir.learning.context import Loss
from xpmir.utils.utils import foreach, easylog

logger = easylog()

PairwiseTriplet = namedtuple("PairwiseTriplet", ["pos_doc", "neg_doc", "query"])


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

    alpha: Param[float] = 0.1
    """The hyperparameter for the KL divergence"""

    def initialize(self):
        decoder_outdim = self.id_generator.decoder_outdim
        alphas = torch.tensor(
            [
                sum(decoder_outdim**i for i in range(j + 1))
                for j in range(self.max_depth, 0, -1)
            ]
        ).to(self.id_generator.device)
        alphas = (1 / alphas).unsqueeze(-1)
        self.kl_target = torch.cat(
            (((1 - alphas) / decoder_outdim).expand(-1, decoder_outdim), alphas), -1
        )
        self.kl_lossfn = nn.KLDivLoss(reduction="batchmean")

    def recursive(
        self,
        decoder_input_tokens,  # None or [bs]
        unfinished_sequences: torch.tensor,  # shape [bs]
        log_cur_node_proba: torch.tensor,  # shape [bs,3]
        depth: int,
        stepwise_generators: PairwiseTriplet,
    ):
        # pass get the probas, each one of shape: [bs, dec_dim+1]
        log_proba = PairwiseTriplet(
            *(g.step(decoder_input_tokens) for g in stepwise_generators)
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("\n")
            logger.debug(f"posdoc_proba: {torch.exp(log_proba.pos_doc)}")
            logger.debug(f"negdoc_proba: {torch.exp(log_proba.neg_doc)}")
            logger.debug(f"query_proba: {torch.exp(log_proba.query)}")

        # middle_term in the formula
        middle_term = torch.sum(
            torch.exp(log_proba.pos_doc.detach() + log_proba.query.detach())
            * (1 - torch.exp(log_proba.neg_doc.detach()))
            * (
                torch.sum(log_cur_node_proba, dim=-1).unsqueeze(-1)
                + log_proba.pos_doc
                + log_proba.query
            ),
            dim=-1,
        )  # shape: [bs]

        # last term in the formula
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

        # randomly choose the target of sampling
        # currently only support the eos_token_id is the last one the decoding dimension
        assert self.id_generator.eos_token_id == log_proba.pos_doc.shape[1] - 1

        # get the bs
        bs = log_cur_node_proba.shape[0]
        sampling_target = torch.randint(low=0, high=3, size=(bs,)).to(
            self.id_generator.device
        )
        # shape [3*bs, dec_dim - 1]
        log_proba_stacks = torch.vstack(
            (
                log_proba.pos_doc[:, :-1],
                log_proba.neg_doc[:, :-1],
                log_proba.query[:, :-1],
            )
        )
        indices = (
            (sampling_target * bs + torch.arange(bs).to(self.id_generator.device))
            .unsqueeze(1)
            .expand(-1, log_proba_stacks.shape[1])
        )
        next_tokens = torch.multinomial(
            torch.exp(torch.gather(log_proba_stacks, 0, indices)), num_samples=1
        ).squeeze(1)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"sampled token is {next_tokens} of depth {depth}")

        # Here we need to use the raw token to calculate
        # to avoid the index out of bound pb (it will be masked anyways)
        # cumulate the proba from root
        iterator_vector = torch.arange(bs)
        # each of shape [bs]
        log_proba_next = PairwiseTriplet(
            *(x[iterator_vector, next_tokens] for x in log_proba)
        )
        log_cur_node_proba = log_cur_node_proba + torch.vstack(
            (
                log_proba_next.pos_doc,
                log_proba_next.neg_doc,
                log_proba_next.query,
            )
        ).transpose(0, 1)

        # mask the generated tokens if some of the seqs
        # are already end before(0 in unfinished_sequences)
        new_unfinished_sequences = (
            next_tokens != self.id_generator.eos_token_id
        ) & unfinished_sequences

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"input token for the next step: {next_tokens}")

        # the kl loss to force all the sequence to have the similar proba
        kl_loss = PairwiseTriplet(
            *(
                self.kl_lossfn(x, self.kl_target[depth].expand(bs, -1))
                for x in log_proba
            )
        )
        kl_loss = kl_loss.pos_doc + kl_loss.neg_doc + kl_loss.query

        # whether need to be end now?
        if new_unfinished_sequences.max() == 0 or depth == self.max_depth - 1:
            return (
                middle_term + last_term - self.alpha * kl_loss
            ) * unfinished_sequences.detach()

        # shape [3, bs]
        sampling_multiplier_stack = torch.vstack(
            (
                log_proba_next.neg_doc.detach() + log_proba_next.query.detach(),
                log_proba_next.pos_doc.detach() + log_proba_next.query.detach(),
                log_proba_next.pos_doc.detach() + log_proba_next.neg_doc.detach(),
            )
        )

        sampling_multiplier = torch.exp(
            sampling_multiplier_stack[sampling_target, iterator_vector]
        )

        return unfinished_sequences.detach() * (
            self.recursive(
                next_tokens,
                new_unfinished_sequences,
                log_cur_node_proba,
                depth + 1,
                stepwise_generators,
            )
            * sampling_multiplier
            + middle_term
            + last_term
            - self.alpha * kl_loss
        )

    def compute(self, records: PairwiseRecords, context: TrainerContext):

        posdocs_text = [pdr.document.get_text() for pdr in records.positives]
        negdocs_text = [ndr.document.get_text() for ndr in records.negatives]
        queries_text = [qr.topic.get_text() for qr in records.unique_queries]

        bs = len(posdocs_text)

        logger.debug("posdocs_text: %s", posdocs_text)
        logger.debug("negdocs_text: %s", negdocs_text)
        logger.debug("queries_text: %s", queries_text)

        # create the generator for the given records,
        # represent the posdoc, negdoc, query, respectively
        stepwise_generators = PairwiseTriplet(
            *[self.id_generator.stepwise_iterator() for _ in range(3)]
        )

        stepwise_generators.pos_doc.init(posdocs_text)
        stepwise_generators.neg_doc.init(negdocs_text)
        stepwise_generators.query.init(queries_text)

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
                0,
                stepwise_generators,
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
