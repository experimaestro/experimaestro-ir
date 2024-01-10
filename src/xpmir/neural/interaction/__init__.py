from typing import Iterable, Optional, List

from experimaestro import Param

from xpmir.neural.dual import (
    DualVectorScorer,
    TopicRecord,
    DocumentRecord,
)
from xpmir.text import TokenizedTextEncoderBase, TokenizerOptions, TokensEncoderOutput

from .common import SimilarityInput, Similarity


class InteractionScorer(DualVectorScorer[SimilarityInput, SimilarityInput]):
    """Interaction-based neural scorer

    This is the base class for all scorers that depend on a map
    of cosine/inner products between query and document token representations.
    """

    encoder: Param[TokenizedTextEncoderBase[str, TokensEncoderOutput]]
    """The embedding model -- the vocab also defines how to tokenize text"""

    query_encoder: Param[
        Optional[TokenizedTextEncoderBase[str, TokensEncoderOutput]]
    ] = None
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
        texts: List[str],
        encoder: TokenizedTextEncoderBase[str, TokensEncoderOutput],
        options: TokenizerOptions,
    ) -> SimilarityInput:
        encoded = encoder(texts)
        return SimilarityInput(encoded.value, encoded.tokenized.mask, options=options)

    def encode_documents(self, records: Iterable[DocumentRecord]) -> SimilarityInput:
        return self.similarity.preprocess(
            self._encode(
                [record.document.get_text() for record in records],
                self.encoder,
                TokenizerOptions(self.dlen),
            )
        )

    def encode_queries(self, records: Iterable[TopicRecord]) -> SimilarityInput:
        return self.similarity.preprocess(
            self._encode(
                [record.topic.get_text() for record in records],
                self._query_encoder,
                TokenizerOptions(self.qlen),
            )
        )
