from typing import List, Tuple, Optional
from attr import define
from datamaestro_text.data.conversation.base import EntryType
import torch
import sys
from experimaestro import Param, Constant
from datamaestro.record import Record
from datamaestro_text.data.ir import TextItem
from datamaestro_text.data.conversation import (
    AnswerEntry,
    ConversationHistoryItem,
)
from xpmir.conversation.learning.reformulation import (
    ConversationRepresentationEncoder,
)
from xpmir.text.encoders import (
    RepresentationOutput,
    TextsRepresentationOutput,
)
from xpmir.letor.trainers.alignment import AlignmentLoss
from xpmir.neural.splade import SpladeTextEncoderV2
from xpmir.utils.logging import easylog

logger = easylog()


@define
class CoSPLADEOutput(RepresentationOutput):
    q_queries: torch.Tensor
    q_answers: torch.Tensor


class AsymetricMSEContextualizedRepresentationLoss(
    AlignmentLoss[CoSPLADEOutput, TextsRepresentationOutput]
):
    """Computes the asymetric loss for CoSPLADE"""

    version: Constant[int] = 2
    """Current version"""

    def __call__(self, input: CoSPLADEOutput, target: TextsRepresentationOutput):
        # Builds up the list of tokens in the gold output
        ids = target.tokenized.ids.cpu()
        sources = []
        tokens = []
        for ix, (ids, length) in enumerate(
            zip(target.tokenized.ids, target.tokenized.lens)
        ):
            for token_id in set(ids[:length]):
                sources.append(ix)
                tokens.append(token_id)

        # Compute the loss
        delta = (
            torch.relu(target.value[sources, tokens] - input.value[sources, tokens])
            ** 2
        )
        return torch.sum(delta) / input.value.numel()


class CoSPLADE(ConversationRepresentationEncoder):
    """CoSPLADE model"""

    history_size: Param[int] = 0
    """Size of history to take into account (0 for infinite)"""

    queries_encoder: Param[SpladeTextEncoderV2[List[List[str]]]]
    """Encoder for the query history (the first one being the current one)"""

    history_encoder: Param[SpladeTextEncoderV2[Tuple[str, str]]]
    """Encoder for (query, answer) pairs"""

    version: Constant[int] = 2
    """Current version"""

    reverse_queries: Param[bool] = False
    """If True, use the order q_n, q_1, ..., q_{n-1}. If False, q_n, q_{n-1}, ..., q_1

    The original CoSPLADE uses True for this parameter
    """

    def __initialize__(self, options):
        super().__initialize__(options)

        self.queries_encoder.initialize(options)
        self.history_encoder.initialize(options)

    @property
    def dimension(self):
        return self.queries_encoder.dimension

    def forward(self, records: List[Record]):
        queries: List[List[str]] = []
        query_answer_pairs: List[Tuple[str, str]] = []
        pair_origins: List[int] = []
        history_size = self.history_size or sys.maxsize

        # Process each topic record

        #: History size for normalization
        history_sizes = torch.zeros((len(records), 1))

        for ix, c_record in enumerate(records):
            # Adds q_n, q_{n-1}, ..., q_{1}
            q_history = [
                entry[TextItem].text
                for entry in c_record[ConversationHistoryItem].history
                if entry[EntryType] == EntryType.USER_QUERY
            ]

            if self.reverse_queries:
                # Adds q_n, q_1, ..., q_{n-1}
                q_history.reverse()

            queries.append([c_record[TextItem].text] + q_history)

            # List of query/answer couples
            answer: Optional[AnswerEntry] = None
            count = 0
            for item in c_record[ConversationHistoryItem].history:
                entry_type = item[EntryType]
                if entry_type == EntryType.USER_QUERY and answer is not None:
                    count += 1
                    query_answer_pairs.append((c_record[TextItem].text, answer.answer))
                    pair_origins.append(ix)
                    if len(pair_origins) >= history_size:
                        break
                elif entry_type == EntryType.SYSTEM_ANSWER:
                    if (answer := item.get(AnswerEntry)) is None:
                        logger.warning("Answer record has no answer entry")
                else:
                    # Ignore anything which is not a pair topic-response
                    answer = None

            history_sizes[ix, 0] = max(count, 1)

        # (1) encodes the queries
        q_queries = self.queries_encoder(queries).value

        # (2) encodes the past queries and answers (if any)
        q_answers = torch.zeros_like(q_queries)
        if query_answer_pairs:
            x_pairs = self.history_encoder(query_answer_pairs).value
            q_ix = torch.tensor(pair_origins, dtype=torch.long, device=q_queries.device)
            q_ix = q_ix.unsqueeze(-1).expand(x_pairs.shape)
            q_answers.scatter_add_(0, q_ix, x_pairs)

            # Normalize by number of pairs
            q_answers /= history_sizes.to(q_queries.device)

        return CoSPLADEOutput(q_queries + q_answers, q_queries, q_answers)
