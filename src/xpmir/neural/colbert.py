# ColBERT implementation
#
# From
# https://github.com/stanford-futuredata/ColBERT/blob/v0.2/colbert/modeling/colbert.py

from experimaestro import Constant, Param, default, Annotated
from torch import nn
from xpmir.text import TokenizerOptions
from xpmir.learning.context import TrainerContext
from xpmir.letor.records import BaseRecords
from xpmir.neural.interaction import InteractionScorer
from .interaction.common import Similarity, CosineSimilarity


class Colbert(InteractionScorer):
    """ColBERT model

    Implementation of the Colbert model from:

        Khattab, Omar, and Matei Zaharia. “ColBERT: Efficient and Effective
        Passage Search via Contextualized Late Interaction over BERT.” SIGIR
        2020, Xi'An, China

    For the standard Colbert model, use BERT as the vocab(ulary)
    """

    version: Constant[int] = 2
    """Current version of the code (changes if a bug is found)"""

    similarity: Annotated[Similarity, default(CosineSimilarity())]
    """Which similarity function to use - ColBERT uses a cosine similarity by default"""

    linear_dim: Param[int] = 128
    """Size of the last linear layer (before computing inner products)"""

    compression_size: Param[int] = 128
    """Projection layer for the last layer (or 0 if None)"""

    def __validate__(self):
        super().__validate__()
        assert not self.encoder.static(), "The vocabulary should be learnable"

        assert self.compression_size >= 0, "Last layer size should be 0 or above"

    def __initialize__(self, options):
        super().__initialize__(options)
        self.linear = nn.Linear(self.encoder.dimension(), self.linear_dim, bias=False)

    def _forward(self, inputs: BaseRecords, info: TrainerContext = None):
        queries = self._query_encoder(
            [qr.topic.get_text() for qr in inputs.queries],
            options=TokenizerOptions(max_length=self.qlen),
        )
        documents = self.encoder(
            [dr.document.get_text() for dr in inputs.documents],
            options=TokenizerOptions(max_length=self.dlen),
        )

        return self.similarity(queries.value, documents.value)


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
    from xpmir.text.huggingface import HFStringTokenizer, HFTokensEncoder
    from xpmir.text.encoders import TokenizedTextEncoder

    assert query_token is False, "Not implemented: use [QUERY] token"
    assert doc_token is False, "Not implemented: use [DOCUMENT] token"
    assert mask_token is False, "Not implemented: use [MASK] token"

    encoder = TokenizedTextEncoder(
        tokenizer=HFStringTokenizer.from_pretrained_id(model_id),
        encoder=HFTokensEncoder.from_pretrained_id(model_id),
    )
    return Colbert(encoder=encoder)
