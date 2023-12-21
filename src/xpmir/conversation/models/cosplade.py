from typing import List, NamedTuple
from attr import dataclass, define
import torch
from experimaestro import Config, Param
from datamaestro_text.data.conversation import (
    Conversation,
    AnswerEntry,
)
from xpmir.learning import Module
from xpmir.conversation.learning.reformulation import (
    ContextualizedRepresentationLoss,
    ConversationRepresentationEncoder,
    ConversationRepresentationOutput,
)
from xpmir.text.encoders import TextListEncoder, DualTextEncoder


@define
class CoSPLADEOutput(ConversationRepresentationOutput):
    q_queries: torch.tensor
    q_answers: torch.Tensor


class AsymetricMSEContextualizedRepresentationLoss(ContextualizedRepresentationLoss):
    """Computes the asymetric loss for CoSPLADE"""

    def __call__(self, input: CoSPLADEOutput, target: ConversationRepresentationOutput):
        return torch.maximum(target.representation - input.q_answers, 0).sum()


class SPLADEQueryEncoder(Module):
    pass


class SPLADEHistoryEncoder(Module):
    pass


class CoSPLADE(ConversationRepresentationEncoder):
    """CoSPLADE model"""

    history_size: Param[int] = 0
    """Size of history to take into account"""

    queries_encoder: Param[SPLADEQueryEncoder]
    """Encoder for the query history (the first one being the current one)"""

    history_encoder: Param[SPLADEHistoryEncoder]
    """Encoder for (query, answer) pairs"""

    def forward(self, records: List[Conversation[AnswerEntry]]):
        # Prepare the data
        queries = []
        query_answer_pairs = []
        pair_origins = []
        for ix, record in enumerate(records):
            queries.append(record.history[-1].query)
            for item, _ in zip(reversed(record.history), self.history_size):
                query_answer_pairs.append((item.query, item.answer))
                pair_origins.append(ix)

        # (1) encodes the queries
        q_queries = self.queries_encoder(queries)

        # (2) encodes the past queries and answers
        x_pairs = self.history_encoder(query_answer_pairs)
        q_answers = torch.zeros_like(q_queries)
        q_answers.scatter_add(0, torch.LongTensor(pair_origins), x_pairs)

        return CoSPLADEOutput(q_queries + q_answers, q_queries, q_answers)
