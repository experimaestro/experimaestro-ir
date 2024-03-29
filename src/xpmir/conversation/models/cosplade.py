from typing import List, Tuple, Optional
from attr import define
from datamaestro_text.data.conversation.base import EntryType
import torch
import sys
from experimaestro import Param
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

        # Compute difference on selected tokens
        difference = torch.nn.functional.mse_loss(
            input.value[sources, tokens],
            target.value[sources, tokens],
            reduction="none",
        )
        loss = torch.zeros(
            len(target.value), dtype=target.value.dtype, device=target.value.device
        )

        # Aggregate
        sources_pt = torch.tensor(sources, device=target.value.device, dtype=torch.long)
        return loss.scatter_add(0, sources_pt, difference).mean()


class CoSPLADE(ConversationRepresentationEncoder):
    """CoSPLADE model"""

    history_size: Param[int] = 0
    """Size of history to take into account (0 for infinite)"""

    queries_encoder: Param[SpladeTextEncoderV2[List[List[str]]]]
    """Encoder for the query history (the first one being the current one)"""

    history_encoder: Param[SpladeTextEncoderV2[Tuple[str, str]]]
    """Encoder for (query, answer) pairs"""

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
        for ix, c_record in enumerate(records):
            # Adds q_n, q_1, ..., q_{n-1}
            queries.append(
                [c_record[TextItem].text]
                + [
                    entry[TextItem].text
                    for entry in c_record[ConversationHistoryItem].history
                    if entry[EntryType] == EntryType.USER_QUERY
                ]
            )

            # List of query/answer couples
            answer: Optional[AnswerEntry] = None
            for item in c_record[ConversationHistoryItem].history:
                entry_type = item[EntryType]
                if entry_type == EntryType.USER_QUERY and answer is not None:
                    query_answer_pairs.append((item[TextItem].text, answer.answer))
                    pair_origins.append(ix)
                    if len(pair_origins) >= history_size:
                        break
                elif entry_type == EntryType.SYSTEM_ANSWER:
                    if (answer := item.get(AnswerEntry)) is None:
                        logger.warning("Answer record has no answer entry")
                else:
                    # Ignore anything which is not a pair topic-response
                    answer = None

        # (1) encodes the queries
        q_queries = self.queries_encoder(queries).value

        # (2) encodes the past queries and answers (if any)
        q_answers = torch.zeros_like(q_queries)
        if query_answer_pairs:
            x_pairs = self.history_encoder(query_answer_pairs).value
            q_ix = torch.tensor(pair_origins, dtype=torch.long, device=q_queries.device)
            q_ix = q_ix.unsqueeze(-1).expand(x_pairs.shape)
            q_answers.scatter_add_(0, q_ix, x_pairs)

        return CoSPLADEOutput(q_queries + q_answers, q_queries, q_answers)
