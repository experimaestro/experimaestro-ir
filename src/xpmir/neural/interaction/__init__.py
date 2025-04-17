from abc import abstractmethod
from typing import Iterable, Optional, List
import torch
from experimaestro import Param, Annotated, field
from xpmir.learning.context import TrainerContext
from xpmir.neural.dual import (
    DualVectorScorer,
    TopicRecord,
    DocumentRecord,
)
from xpmir.text import TokenizedTextEncoderBase, TokenizerOptions, TokensEncoderOutput
from xpmir.text.encoders import InputType
from .common import SimilarityInput, SimilarityOutput, Similarity


class InteractionScorer(DualVectorScorer[SimilarityInput, SimilarityInput]):
    """Interaction-based neural scorer

    This is the base class for all scorers that depend on a map
    of cosine/inner products between query and document token representations.
    """

    encoder: Param[TokenizedTextEncoderBase[InputType, TokensEncoderOutput]]
    """The embedding model -- the vocab also defines how to tokenize text"""

    query_encoder: Annotated[
        Optional[TokenizedTextEncoderBase[InputType, TokensEncoderOutput]],
        field(default=None),
    ]
    """The embedding model for queries (if None, uses encoder)"""

    similarity: Param[Similarity]
    """Which similarity function to use - ColBERT uses a cosine similarity by default"""

    qlen: Param[int] = 20
    """Maximum query length (this can be even shortened by the model)"""

    dlen: Param[int] = 2000
    """Maximum document length (this can be even shortened by the model)"""

    def __validate__(self):
        super().__validate__()
        assert (
            self.dlen <= self.encoder.max_tokens()
        ), f"The maximum document length ({self.dlen}) should be less "
        "that what the vocab can process ({self.encoder.max_tokens()})"
        assert (
            self.qlen <= self.encoder.max_tokens()
        ), f"The maximum query length ({self.qlen}) should be less "
        "that what the vocab can process ({self.encoder.max_tokens()})"

    def _encode(
        self,
        texts: List[InputType],
        encoder: TokenizedTextEncoderBase[InputType, TokensEncoderOutput],
        options: TokenizerOptions,
    ) -> SimilarityInput:
        encoded = encoder(texts, options=options)
        return self.similarity.preprocess(
            SimilarityInput(encoded.value, encoded.tokenized.mask)
        )

    def encode_documents(self, records: Iterable[DocumentRecord]) -> SimilarityInput:
        return self.similarity.preprocess(
            self._encode(
                records,
                self.encoder,
                TokenizerOptions(self.dlen),
            )
        )

    def encode_queries(self, records: Iterable[TopicRecord]) -> SimilarityInput:
        return self.similarity.preprocess(
            self._encode(
                records,
                self._query_encoder,
                TokenizerOptions(self.qlen),
            )
        )

    def score_pairs(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
        info: Optional[TrainerContext] = None,
    ) -> torch.Tensor:
        similarity = self.similarity.compute_pairs(queries, documents)
        return self.compute_scores(queries, documents, similarity, info)

    def score_product(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
        info: Optional[TrainerContext] = None,
    ) -> torch.Tensor:
        similarity = self.similarity.compute_product(queries, documents)
        return self.compute_scores(queries, documents, similarity, info)

    @abstractmethod
    def compute_scores(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
        similarity: SimilarityOutput,
        info: Optional[TrainerContext] = None,
    ):
        ...
