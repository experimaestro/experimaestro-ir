from experimaestro import Param
from experimaestro.compat import cached_property
from typing import List, NamedTuple, Optional
from collections import namedtuple
from abc import abstractmethod

import torch
import torch.nn as nn
import numpy as np
from xpmir.learning.optim import Module
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.rankers import AbstractModuleScorer
from xpmir.utils.utils import easylog

logger = easylog()

PairwiseTuple = namedtuple("PairwiseTuple", ["doc", "qry"])


class GeneratorForwardOutput(NamedTuple):
    """The forward output of the generative retrieval"""

    logits: torch.tensor
    past_key_values: Optional[torch.tensor] = None


class StepwiseGenerator:
    """Utility class for generating one token at a time"""

    decoder_input_ids: torch.LongTensor
    """The current token"""

    @abstractmethod
    def init(self, texts: List[str]) -> torch.Tensor:
        """Returns the distribution over the first generated tokens (BxV)"""
        pass

    @abstractmethod
    def step(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """Returns the distribution over next tokens (BxV), given the last
        generates ones (B)"""
        pass


class IdentifierGenerator(Module):
    """Models that generate an identifier given a document or a query"""

    def __initialize__(self):
        pass

    @abstractmethod
    def stepwise_iterator(self) -> StepwiseGenerator:
        pass


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


class GenerativeRetrievalScorer(AbstractModuleScorer):
    """The abstract class for the generative retrieval scorer"""

    id_generator: Param[IdentifierGenerator]
    """The id generator"""

    start_max_depth: Param[int] = -1
    """if apply progressive training, the starter max depth. If it is a negative
    number, means we dont't apply progressive training """

    max_depth: Param[int] = 5
    """The max depth we need to consider"""

    def _initialize(self, random):
        self.id_generator.initialize()
        self.current_max_depth = (
            self.start_max_depth if self.start_max_depth > 0 else self.max_depth
        )

    def update_depth(self):
        if self.current_max_depth < self.max_depth:
            self.current_max_depth += 1
            logger.info(
                f"Update the depth for scorer: current depth is {self.current_max_depth}"
            )


class NaiveGenerativeRetrievalScorer(GenerativeRetrievalScorer):
    """A naive scorer which will be used for the inference of the generative retrieval model,
    and this scorer is not learnable"""

    early_finish_punishment: Param[int] = 1
    """A early finish punishment hyperparameter, trying to make model score less
    if the id list is too short. Default value to 1 means no punishment"""

    def recursive(
        self,
        decoder_input_tokens,  # shape [bs]
        unfinished_sequences,
        depth,
        stepwise_generators: PairwiseTuple,
    ):
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

        if new_unfinished_sequences.max() == 0 or depth == self.current_max_depth - 1:
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
            decoder_input_tokens, unfinished_sequences, 0, stepwise_generator
        )


class RandomBasedGenerativeRetrievalScorer(GenerativeRetrievalScorer):
    """A scorer based on the probability that a document is better than a random
    document given a query, and this scorer is not learnable"""

    @cached_property
    def random_distribution(self):
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
        depth,
        stepwise_generators: PairwiseTuple,
    ):
        # pass get the probas
        logits = PairwiseTuple(
            *[g.step(decoder_input_tokens) for g in stepwise_generators]
        )
        log_proba = PairwiseTuple(
            *(nn.functional.log_softmax(logit, dim=-1) for logit in logits)
        )

        log_proba_randdoc = self.random_distribution[depth]
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

        if new_unfinished_sequences.max() == 0 or depth == self.current_max_depth - 1:
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
            decoder_input_tokens, unfinished_sequences, 0, stepwise_generator
        )
