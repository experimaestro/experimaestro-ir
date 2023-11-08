from experimaestro import Param
from typing import List
from abc import abstractmethod

import torch

from xpmir.learning.optim import Module
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.rankers import AbstractModuleScorer


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


class GenerativeRetrievalScorer(AbstractModuleScorer):
    """A scorer which will be used for the inference of the generative retrieval model,
    and this scorer is not learnable"""

    id_generator: Param[IdentifierGenerator]
    """The id generator"""

    early_finish_punishment: Param[int] = 1
    """A early finish punishment hyperparameter, trying to make model score less
    if the id list is too short. Default value to 1 means no punishment"""

    max_depth: Param[int] = 5
    """The max depth we need to consider"""

    def _initialize(self, random):
        self.id_generator.initialize()

    def recursive(
        self,
        decoder_input_tokens,
        unfinished_sequences,
        depth,
        doc_stepwise_generator: StepwiseGenerator,
        qry_stepwise_generator: StepwiseGenerator,
    ):
        # pass get the probas
        log_doc_proba = doc_stepwise_generator.step(decoder_input_tokens)
        log_qry_proba = qry_stepwise_generator.step(decoder_input_tokens)

        # sampling according to the proba distribution --> shape bs
        raw_next_tokens = torch.multinomial(
            torch.exp(log_qry_proba), num_samples=1
        ).squeeze(1)

        iterator_vector = torch.arange(len(raw_next_tokens))
        log_doc_proba_next_tokens = log_doc_proba[iterator_vector, raw_next_tokens]
        log_qry_proba_next_tokens = log_qry_proba[iterator_vector, raw_next_tokens]

        # mask them! For the sequence already finished, replaced by the
        # multiplier 1(or some other multiplier to punish the early finish)
        log_doc_proba_next_tokens[
            unfinished_sequences == 0
        ] = self.early_finish_punishment
        log_qry_proba_next_tokens[
            unfinished_sequences == 0
        ] = self.early_finish_punishment

        # mask the generated tokens if some of the seqs
        # are already end before(0 in unfinished_sequences)
        raw_next_tokens = (
            raw_next_tokens * unfinished_sequences
            + self.id_generator.pad_token_id * (1 - unfinished_sequences)
        )
        decoder_input_tokens = raw_next_tokens.unsqueeze(-1)
        new_unfinished_sequences = unfinished_sequences.mul(
            raw_next_tokens.tile(1, 1)
            .ne(
                torch.tensor([self.id_generator.eos_token_id]).to(
                    self.id_generator.device
                )
            )
            .prod(dim=0)
        )

        if new_unfinished_sequences.max() == 0 or depth == self.max_depth:
            return torch.exp(log_doc_proba_next_tokens)

        return self.recursive(
            decoder_input_tokens,
            new_unfinished_sequences,
            depth + 1,
            doc_stepwise_generator,
            qry_stepwise_generator,
        ) * torch.exp(log_doc_proba_next_tokens)

    def forward(
        self, inputs: "BaseRecords", info: TrainerContext = None
    ):  # try to return tensor [bs, ] which contains the scores
        # also implemented in a recursive way
        queries_text = [pdr.topic.get_text() for pdr in inputs.topics]
        documents_text = [ndr.document.get_text() for ndr in inputs.documents]

        bs = len(queries_text)

        doc_stepwise_generator = self.id_generator.stepwise_iterator()
        qry_stepwise_generator = self.id_generator.stepwise_iterator()

        qry_stepwise_generator.init(queries_text)
        doc_stepwise_generator.init(documents_text)

        # initialization
        decoder_input_tokens = None
        unfinished_sequences = torch.ones(bs, dtype=torch.long).to(
            self.id_generator.device
        )

        return self.recursive(
            decoder_input_tokens,
            unfinished_sequences,
            1,
            doc_stepwise_generator,
            qry_stepwise_generator,
        )
