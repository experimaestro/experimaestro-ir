from typing import List, Optional
from experimaestro import Config, Param
import torch.nn.functional as F
import torch.nn as nn
import torch
from xpmir.distributed import DistributableModel
from xpmir.text.huggingface import (
    OneHotHuggingFaceEncoder,
    TransformerTokensEncoderWithMLMOutput,
)
from xpmir.text.encoders import TextEncoder
from xpmir.neural.dual import DotDense, ScheduledFlopsRegularizer
from xpmir.utils.utils import easylog

logger = easylog()


class Aggregation(Config):
    """The aggregation function for Splade"""

    def with_linear(self, logits, mask, weight, bias=None):
        """Project before aggregating using a linear transformation

        Can be optimized by further operators

        :param logits: The logits output by the sequence representation model (B
            x L x D)
        :param mask: The mask (B x L) where 0 when the element should be masked
            out
        :param weight: The linear transformation (D' x D)
        """
        projection = F.linear(logits, weight, bias)
        return self(projection, mask)


class MaxAggregation(Aggregation):
    """Aggregate using a max"""

    def __call__(self, logits, mask):
        # Get the maximum (masking the values)
        values, _ = torch.max(
            torch.relu(logits) * mask.to(logits.device).unsqueeze(-1),
            dim=1,
        )

        # Computes log(1+x)
        return torch.log1p(values.clamp(min=0))


class SumAggregation(Aggregation):
    """Aggregate using a sum"""

    def __call__(self, logits, mask):
        return torch.sum(
            torch.log1p(torch.relu(logits) * mask.to(logits.device).unsqueeze(-1)),
            dim=1,
        )


class SpladeTextEncoderModel(nn.Module):
    def __init__(self, encoder, aggregation):
        super().__init__()
        self.encoder = encoder
        self.aggregation = aggregation

    def forward(self, tokenized):
        # We stock all the outputs in order to get the embedding matrix
        # Here as the automodel is not the same as the normal AutoModel,
        # So here the output has the attribute logits, the w_ij in the paper
        # which is of shape (bs, len(texts), vocab_size)
        out = self.encoder(tokenized, all_outputs=True)
        out = self.aggregation(out.logits, tokenized.mask)
        return out


class SpladeTextEncoder(TextEncoder, DistributableModel):
    """Splade model

    It is only a text encoder since the we use `xpmir.neural.dual.DotDense`
    as the scorer class
    """

    encoder: Param[TransformerTokensEncoderWithMLMOutput]
    """The encoder from Hugging Face"""

    aggregation: Param[Aggregation]
    """How to aggregate the vectors"""

    maxlen: Param[Optional[int]] = None
    """Max length for texts"""

    def __initialize__(self, random=None):
        self.encoder.initialize()
        self.model = SpladeTextEncoderModel(self.encoder, self.aggregation)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a batch x vocab tensor"""
        tokenized = self.encoder.batch_tokenize(texts, mask=True, maxlen=self.maxlen)
        out = self.model(tokenized)
        return out

    @property
    def dimension(self):
        return self.encoder.model.config.vocab_size

    def static(self):
        return False

    def distribute_models(self, update):
        self.model = update(self.model)


def _splade(
    lambda_q: float,
    lambda_d: float,
    aggregation: Aggregation,
    lamdba_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    # Unlike the cross-encoder, here the encoder returns the whole last layer
    # In the paper we use the DistilBERT-based as the checkpoint
    encoder = TransformerTokensEncoderWithMLMOutput(model_id=hf_id, trainable=True)

    # make use the output of the BERT and do an aggregation
    doc_encoder = SpladeTextEncoder(
        aggregation=aggregation, encoder=encoder, maxlen=200
    )
    query_encoder = SpladeTextEncoder(
        aggregation=aggregation, encoder=encoder, maxlen=30
    )

    return DotDense(
        encoder=doc_encoder, query_encoder=query_encoder
    ), ScheduledFlopsRegularizer(
        lambda_q=lambda_q,
        lambda_d=lambda_d,
        lamdba_warmup_steps=lamdba_warmup_steps,
    )


def _splade_doc(
    lambda_q: float,
    lambda_d: float,
    aggregation: Aggregation,
    lamdba_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    # Unlike the cross-encoder, here the encoder returns the whole last layer
    # The doc_encoder is the traditional one, and the query encoder return a vector
    # contains only 0 and 1
    # In the paper we use the DistilBERT-based as the checkpoint
    encoder = TransformerTokensEncoderWithMLMOutput(model_id=hf_id, trainable=True)

    # make use the output of the BERT and do an aggregation
    doc_encoder = SpladeTextEncoder(
        aggregation=aggregation, encoder=encoder, maxlen=256
    )

    query_encoder = OneHotHuggingFaceEncoder(model_id=hf_id, maxlen=30)

    return DotDense(
        encoder=doc_encoder, query_encoder=query_encoder
    ), ScheduledFlopsRegularizer(
        lambda_q=lambda_q,
        lambda_d=lambda_d,
        lamdba_warmup_steps=lamdba_warmup_steps,
    )


def spladeV1(
    lambda_q: float,
    lambda_d: float,
    lamdba_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    """Returns the Splade architecture"""
    return _splade(lambda_q, lambda_d, SumAggregation(), lamdba_warmup_steps, hf_id)


def spladeV2_max(
    lambda_q: float,
    lambda_d: float,
    lamdba_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    """Returns the Splade-max architecture

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval
    (arXiv:2109.10086)
    """
    return _splade(lambda_q, lambda_d, MaxAggregation(), lamdba_warmup_steps, hf_id)


def spladeV2_doc(
    lambda_q: float,
    lambda_d: float,
    lamdba_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    """Returns the Splade-doc architecture

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval
    (arXiv:2109.10086)
    """
    return _splade_doc(lambda_q, lambda_d, MaxAggregation(), lamdba_warmup_steps, hf_id)
