# Pre-trained models

from xpmir.neural.dual import DotDense
from xpmir.text.huggingface import TransformerTextEncoderAdapter


def tas_balanced():
    """Returns the TAS-Balanced model (from huggingface)

    > Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling
    > Sebastian Hofstätter, Sheng-Chieh Lin, Jheng-Hong Yang, Jimmy Lin, Allan Hanbury

    Returns:
        DotDense: A DotDense ranker based on tas-balanced
    """
    from xpmir.text.huggingface import TransformerEncoder

    encoder = TransformerEncoder(
        model_id="sebastian-hofstaetter/distilbert-dot-tas_b-b256-msmarco",
        trainable=True,
        maxlen=200,
    )

    return DotDense(
        encoder=encoder,
        query_encoder=TransformerTextEncoderAdapter(encoder=encoder, maxlen=30),
    )


def spladev2() -> DotDense:
    """The Splade V2 model (from https://github.com/naver/splade)

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval,
    Thibault Formal, Benjamin Piwowarski, Carlos Lassance, and Stéphane Clinchant.

    https://arxiv.org/abs/2109.10086
    """
    pass
