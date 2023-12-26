from typing import List, Tuple
from attr import define
import torch
from experimaestro import Param
from xpmir.conversation.records import HistoryRecord
from xpmir.conversation.learning.reformulation import (
    ContextualizedRepresentationLoss,
    ConversationRepresentationEncoder,
    ConversationRepresentationOutput,
)
from xpmir.neural.splade import SpladeTextEncoderV2


@define
class CoSPLADEOutput(ConversationRepresentationOutput):
    q_queries: torch.tensor
    q_answers: torch.Tensor


class AsymetricMSEContextualizedRepresentationLoss(ContextualizedRepresentationLoss):
    """Computes the asymetric loss for CoSPLADE"""

    def __call__(self, input: CoSPLADEOutput, target: ConversationRepresentationOutput):
        return torch.maximum(target.representation - input.q_answers, 0).sum()


class CoSPLADE(ConversationRepresentationEncoder):
    """CoSPLADE model"""

    history_size: Param[int] = 0
    """Size of history to take into account"""

    queries_encoder: Param[SpladeTextEncoderV2]
    """Encoder for the query history (the first one being the current one)"""

    history_encoder: Param[SpladeTextEncoderV2]
    """Encoder for (query, answer) pairs"""

    def forward(self, records: List[HistoryRecord]):
        queries: List[List[str]] = []
        query_answer_pairs: List[Tuple[str, str]] = []
        pair_origins: List[int] = []

        for ix, record in enumerate(records):
            # Adds q_n, q_1, ..., q_{n-1}
            queries.append(
                [record.topic.get_text()]
                + [topic.get_text() for topic in record.history]
            )

            # List of query/answer couples
            for item, _ in zip(reversed(record.history), range(self.history_size)):
                query_answer_pairs.append((item.query, item.answer))
                pair_origins.append(ix)

        # (1) encodes the queries
        q_queries = self.queries_encoder(queries)

        # (2) encodes the past queries and answers
        x_pairs = self.history_encoder(query_answer_pairs)
        q_answers = torch.zeros_like(q_queries)
        q_answers.scatter_add(0, torch.LongTensor(pair_origins), x_pairs)

        return CoSPLADEOutput(q_queries + q_answers, q_queries, q_answers)
