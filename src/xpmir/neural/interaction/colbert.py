from typing import List, Optional
from experimaestro import Constant, Param
from torch import nn
from xpmir.text import (
    TokensEncoderOutput,
    TokenizedTextEncoderBase,
    TokenizerOptions,
)
from xpmir.text.encoders import InputType
from xpmir.neural.interaction import (
    InteractionScorer,
    SimilarityInput,
    SimilarityOutput,
    TrainerContext,
)
from xpmir.neural.interaction.common import CosineSimilarity


class Colbert(InteractionScorer):
    """ColBERT model

    Implementation of the Colbert model from:

        Khattab, Omar, and Matei Zaharia. “ColBERT: Efficient and Effective
        Passage Search via Contextualized Late Interaction over BERT.” SIGIR
        2020, Xi'An, China

    For the standard Colbert model, use the colbert function
    """

    version: Constant[int] = 2
    """Current version of the code (changes if a bug is found)"""

    linear_dim: Param[int] = 128
    """Size of the last linear layer (before computing inner products)"""

    compression_size: Param[int] = 128
    """Projection layer for the last layer (or 0 if None)"""

    def __validate__(self):
        super().__validate__()

        assert self.compression_size >= 0, "Last layer size should be 0 or above"

    def __initialize__(self, options):
        super().__initialize__(options)
        self.linear = nn.Linear(self.encoder.dimension, self.linear_dim, bias=False)

    def _encode(
        self,
        texts: List[InputType],
        encoder: TokenizedTextEncoderBase[InputType, TokensEncoderOutput],
        options: TokenizerOptions,
    ) -> SimilarityInput:
        encoded = encoder(texts, options=options)
        return self.similarity.preprocess(
            SimilarityInput(self.linear(encoded.value), encoded.tokenized.mask)
        )

    def compute_scores(
        self,
        queries: SimilarityInput,
        documents: SimilarityInput,
        value: SimilarityOutput,
        info: Optional[TrainerContext] = None,
    ):
        # Similarity matrix B x Lq x Ld or Bq x Lq x Bd x Ld
        s = value.similarity.masked_fill(
            value.d_view(documents.mask).logical_not(), float("-inf")
        ).masked_fill(value.q_view(queries.mask).logical_not(), 0)
        return s.max(-1).values.sum(1).flatten()


def colbert(
    model_id: str,
    *,
    query_token: bool = False,
    doc_token: bool = False,
    mask_token: bool = True,
):
    """Creates standard ColBERT model based on a HuggingFace transformer

    :param model_id: The HF model ID
    :param query_token: Whether to use a query prefix token when encoding
        queries, defaults to False
    :param doc_token: Whether to use a document prefix token to encode
        documents, defaults to False
    :param mask_token: Whether to use a mask tokens to encode queries, defaults
        to True
    :return: A ColBERT configuration object
    """
    from xpmir.text.encoders import TokenizedTextEncoder
    from xpmir.text.adapters import TopicTextConverter
    from xpmir.text.huggingface import HFTokensEncoder
    from xpmir.text.huggingface.tokenizers import HFTokenizerAdapter

    assert query_token is False, "Not implemented: use [QUERY] token"
    assert doc_token is False, "Not implemented: use [DOCUMENT] token"
    assert mask_token is False, "Not implemented: use [MASK] token"

    encoder = TokenizedTextEncoder(
        tokenizer=HFTokenizerAdapter.from_pretrained_id(
            model_id,
            converter=TopicTextConverter(),
        ),
        encoder=HFTokensEncoder.from_pretrained_id(model_id),
    )
    return Colbert(encoder=encoder, similarity=CosineSimilarity())
