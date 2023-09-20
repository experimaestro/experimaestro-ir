from experimaestro import Param
from typing import Optional, List
from abc import abstractmethod

import torch
from torch import nn
import numpy as np

from xpmir.learning.optim import Module
from xpmir.letor.records import TokenizedTexts, BaseRecords
from xpmir.distributed import DistributableModel
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

    # FIXME: use rather a flag (it can be either or a document)
    sampling_target: Param[str] = "qry"
    """sampling according to which target during the scoring"""

    def __initialize__(self):
        # FIXME: shouldn't be here / the stepwise generator is only used when computing the score, i.e. in
        # forward (and then those should be arguments for recursive)
        # Load the state_dict?
        self.doc_stepwise_generator = self.id_generator.stepwise_iterator()
        self.qry_stepwise_generator = self.id_generator.stepwise_iterator()

    # FIXME: no need for sampling_target since self.sampling_target gives access to it
    def recursive(self, sampling_target):
        # pass get the probas
        doc_proba = self.doc_stepwise_generator.step()
        qry_proba = self.qry_stepwise_generator.step()

        # obtain the previous unfinished sequence as a mask
        # 0 means no need to continue
        unfinished_sequences = self.doc_stepwise_generator.get_token_state()[1]

        # sampling according to the proba distribution
        if sampling_target == "doc":
            raw_next_tokens = torch.multinomial(doc_proba, num_samples=1).squeeze(
                1
            )  # shape bs
        elif sampling_target == "qry":
            raw_next_tokens = torch.multinomial(qry_proba, num_samples=1).squeeze(
                1
            )  # shape bs
        else:
            raise ValueError()

        # mask the generated tokens if some of the seqs
        # are already end before(0 in unfinished_sequences)
        self.doc_stepwise_generator.set_token_state(raw_next_tokens)
        self.qry_stepwise_generator.set_token_state(raw_next_tokens)

        # get the processed tokens
        next_tokens = self.doc_stepwise_generator.get_token_state()[0].squeeze(-1)

        iterator_vector = torch.arange(len(next_tokens))
        doc_proba_next_tokens = doc_proba[iterator_vector, next_tokens]
        qry_proba_next_tokens = qry_proba[iterator_vector, next_tokens]

        # mask them! For the sequence already finished, replaced by the
        # multiplier 1(or some other multiplier to punish the early finish)
        doc_proba_next_tokens[unfinished_sequences == 0] = self.early_finish_punishment
        qry_proba_next_tokens[unfinished_sequences == 0] = self.early_finish_punishment

        if self.doc_stepwise_generator.stopping_criteria():
            return (
                doc_proba_next_tokens
                if sampling_target == "qry"
                else qry_proba_next_tokens
            )

        if sampling_target == "doc":
            sampling_multiplier = qry_proba_next_tokens
        elif sampling_target == "qry":
            sampling_multiplier = doc_proba_next_tokens
        else:
            raise ValueError()

        return self.recursive(sampling_target) * sampling_multiplier

    def forward(
        self, inputs: "BaseRecords"
    ):  # try to return tensor [bs, ] which contains the scores
        # also implemented in a recursive way
        queries_text = [pdr.topic.get_text() for pdr in inputs.topics]
        documents_text = [ndr.document.get_text() for ndr in inputs.documents]

        # FIXME: shouldn't be using self (see remarks above)
        self.qry_stepwise_generator.init(queries_text)
        self.doc_stepwise_generator.init(documents_text)

        return self.recursive(self.sampling_target)
