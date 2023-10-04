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

    @abstractmethod
    def set_token_state(self, new_tokens: torch.LongTensor):
        """Update the token state for the next step's generation"""
        pass

    @abstractmethod
    def get_token_state(self):
        """Return the token state, including the token, mask, etc"""
        pass

    @abstractmethod
    def stopping_criteria(self) -> bool:
        pass


class IdentifierGenerator(Module):
    """Models that generate an identifier given a document or a query"""

    hf_id: Param[str]
    """The HuggingFace identifier (to configure the model)"""

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

    def _initialize(self, random):
        self.id_generator.initialize()

    def recursive(
        self,
        doc_stepwise_generator: StepwiseGenerator,
        qry_stepwise_generator: StepwiseGenerator,
    ):
        # pass get the probas
        doc_proba = doc_stepwise_generator.step()
        qry_proba = qry_stepwise_generator.step()

        # obtain the previous unfinished sequence as a mask
        # 0 means no need to continue
        unfinished_sequences = doc_stepwise_generator.get_token_state()[1]

        # sampling according to the proba distribution --> shape bs
        raw_next_tokens = torch.multinomial(qry_proba, num_samples=1).squeeze(1)

        # mask the generated tokens if some of the seqs
        # are already end before(0 in unfinished_sequences)
        doc_stepwise_generator.set_token_state(raw_next_tokens)
        qry_stepwise_generator.set_token_state(raw_next_tokens)

        # get the processed tokens
        next_tokens = doc_stepwise_generator.get_token_state()[0].squeeze(-1)

        iterator_vector = torch.arange(len(next_tokens))
        doc_proba_next_tokens = doc_proba[iterator_vector, next_tokens]
        qry_proba_next_tokens = qry_proba[iterator_vector, next_tokens]

        # mask them! For the sequence already finished, replaced by the
        # multiplier 1(or some other multiplier to punish the early finish)
        doc_proba_next_tokens[unfinished_sequences == 0] = self.early_finish_punishment
        qry_proba_next_tokens[unfinished_sequences == 0] = self.early_finish_punishment

        if doc_stepwise_generator.stopping_criteria():
            return doc_proba_next_tokens

        return (
            self.recursive(doc_stepwise_generator, qry_stepwise_generator)
            * doc_proba_next_tokens
        )

    def forward(
        self, inputs: "BaseRecords", info: TrainerContext = None
    ):  # try to return tensor [bs, ] which contains the scores
        # also implemented in a recursive way
        queries_text = [pdr.topic.get_text() for pdr in inputs.topics]
        documents_text = [ndr.document.get_text() for ndr in inputs.documents]

        doc_stepwise_generator = self.id_generator.stepwise_iterator()
        qry_stepwise_generator = self.id_generator.stepwise_iterator()

        qry_stepwise_generator.init(queries_text)
        doc_stepwise_generator.init(documents_text)

        return self.recursive(doc_stepwise_generator, qry_stepwise_generator)
