from typing import Optional, Generic
from experimaestro import Config, Param
import torch.nn as nn
import torch
from xpmir.text.huggingface.encoders import OneHotHuggingFaceEncoder
from xpmir.text import TokenizerOptions
from xpmir.text.huggingface import HFTokenizerBase
from xpmir.text.huggingface.tokenizers import HFTokenizer, HFTokenizerAdapter
from xpmir.text.adapters import TopicTextConverter
from xpmir.text.encoders import (
    TextEncoderBase,
    InputType as EncoderInputType,
    TextsRepresentationOutput,
)
from xpmir.neural.dual import DotDense, ScheduledFlopsRegularizer
from xpmir.text.huggingface.base import (
    HFConfigID,
    HFMaskedLanguageModel,
    HFModelInitFromID,
)
from xpm_torch.module import initialized
import logging

logger = logging.getLogger(__name__)


class Aggregation(Config):
    """The aggregation function for SPLADE"""

    def get_output_module(self, linear: nn.Module) -> nn.Module:
        return AggregationModule(linear, self)


class MaxAggregation(Aggregation):
    """Aggregate using a max"""

    def __call__(self, logits, mask):
        assert logits.shape[:2] == mask.shape[:2], (
            f"Shape mismatch: logits {logits.shape} vs mask {mask.shape}"
        )
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
        assert logits.shape[:2] == mask.shape[:2], (
            f"Shape mismatch: logits {logits.shape} vs mask {mask.shape}"
        )
        return torch.sum(
            torch.log1p(torch.relu(logits) * mask.to(logits.device).unsqueeze(-1)),
            dim=1,
        )


class AggregationModule(nn.Module):
    """The aggregation module for SPLADE, which applies a linear layer followed
    by an aggregation"""

    def __init__(self, linear: nn.Linear, aggregation: Aggregation):
        super().__init__()
        self.linear = linear
        self.aggregation = aggregation

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        return self.aggregation(self.linear(input), mask)


class IdentityWithBias(nn.Identity):
    def __init__(self, original_linear: nn.Linear = None):
        # So that set_output_embeddings is happy
        super().__init__()
        self.bias = None
        self.original_linear = original_linear


class SpladeTextEncoder(
    TextEncoderBase[EncoderInputType, TextsRepresentationOutput],
    Generic[EncoderInputType],
):
    """Splade model text encoder

    It is only a text encoder since the we use `xpmir.neural.dual.DotDense` as
    the scorer class. Compared to V1, it uses the new text HF encoder
    abstractions.
    """

    tokenizer: Param[HFTokenizerBase[EncoderInputType]]
    """The tokenizer from Hugging Face"""

    encoder: Param[HFMaskedLanguageModel]
    """The encoder from Hugging Face"""

    aggregation: Param[Aggregation]
    """How to aggregate the vectors"""

    maxlen: Param[Optional[int]] = None
    """Max length for texts"""

    def __initialize__(self):
        """Module initialization: initializes the encoder and tokenizer, and
        adds the aggregation head."""

        self.encoder.initialize()
        self.tokenizer.initialize()

        # Adds the aggregation head right away - this could allow
        # optimization e.g. for a top-k max aggregation method.

        # Note: ideally we should not modify the encoder in-place, so that
        # the HFMaskedLanguageModel remains reusable.

        # When the encoder is shared between doc/query encoders, the second
        # SpladeTextEncoder finds IdentityWithBias already in place — in that
        # case retrieve the stored original linear.
        output_embeddings = self.encoder.model.get_output_embeddings()
        if isinstance(output_embeddings, IdentityWithBias):
            # Shared encoder: output embeddings already replaced; reuse original
            original_linear = output_embeddings.original_linear
        else:
            assert isinstance(output_embeddings, nn.Linear), (
                f"Cannot handle output embeddings of class {output_embeddings.__class__}"
            )
            original_linear = output_embeddings
            self.encoder.model.set_output_embeddings(
                IdentityWithBias(original_linear=original_linear)
            )

        self.aggregation = self.aggregation.get_output_module(original_linear)

    @initialized
    def forward(self, texts: EncoderInputType) -> TextsRepresentationOutput:
        """Returns a batch x vocab tensor"""
        tokenized = self.tokenizer.tokenize(
            texts, options=TokenizerOptions(self.maxlen)
        )

        value = self.aggregation(self.encoder(tokenized).logits, tokenized.mask)
        return TextsRepresentationOutput(value, tokenized)

    @property
    def dimension(self):
        return self.encoder.model.config.vocab_size

    def static(self):
        return False


def _splade(
    lambda_q: float,
    lambda_d: float,
    aggregation: Aggregation,
    lambda_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    # Unlike the cross-encoder, here the encoder returns the whole last layer
    # In the paper we use the DistilBERT-based as the checkpoint
    encoder = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=hf_id))
    init_hf = HFModelInitFromID.C(model=encoder)
    tokenizer = HFTokenizerAdapter.C(
        tokenizer=HFTokenizer.C(model_id=hf_id),
        converter=TopicTextConverter.C(),
    )

    # make use the output of the BERT and do an aggregation
    doc_encoder = SpladeTextEncoder.C(
        aggregation=aggregation, encoder=encoder, tokenizer=tokenizer, maxlen=200
    )
    query_encoder = SpladeTextEncoder.C(
        aggregation=aggregation, encoder=encoder, tokenizer=tokenizer, maxlen=30
    )

    return (
        DotDense.C(encoder=doc_encoder, query_encoder=query_encoder),
        ScheduledFlopsRegularizer.C(
            lambda_q=lambda_q,
            lambda_d=lambda_d,
            lambda_warmup_steps=lambda_warmup_steps,
        ),
        [init_hf],
    )


def _splade_doc(
    lambda_q: float,
    lambda_d: float,
    aggregation: Aggregation,
    lambda_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    # Unlike the cross-encoder, here the encoder returns the whole last layer
    # The doc_encoder is the traditional one, and the query encoder return a vector
    # contains only 0 and 1
    # In the paper we use the DistilBERT-based as the checkpoint
    encoder = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=hf_id))
    init_hf = HFModelInitFromID.C(model=encoder)
    tokenizer = HFTokenizerAdapter.C(
        tokenizer=HFTokenizer.C(model_id=hf_id),
        converter=TopicTextConverter.C(),
    )
    doc_encoder = SpladeTextEncoder.C(
        aggregation=aggregation, encoder=encoder, tokenizer=tokenizer, maxlen=200
    )

    query_encoder = OneHotHuggingFaceEncoder.C(model_id=hf_id, maxlen=30)

    return (
        DotDense.C(encoder=doc_encoder, query_encoder=query_encoder),
        ScheduledFlopsRegularizer.C(
            lambda_q=lambda_q,
            lambda_d=lambda_d,
            lambda_warmup_steps=lambda_warmup_steps,
        ),
        [init_hf],
    )


def spladeV1(
    lambda_q: float,
    lambda_d: float,
    lambda_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    """Returns the Splade architecture

    :returns: (model, regularizer, init_tasks) tuple
    """
    return _splade(lambda_q, lambda_d, SumAggregation.C(), lambda_warmup_steps, hf_id)


def spladeV2_max(
    lambda_q: float,
    lambda_d: float,
    lambda_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    """Returns the Splade-max architecture

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval
    (arXiv:2109.10086)

    :returns: (model, regularizer, init_tasks) tuple
    """
    return _splade(lambda_q, lambda_d, MaxAggregation.C(), lambda_warmup_steps, hf_id)


def spladeV2_doc(
    lambda_q: float,
    lambda_d: float,
    lambda_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    """Returns the Splade-doc architecture

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval
    (arXiv:2109.10086)

    :returns: (model, regularizer, init_tasks) tuple
    """
    return _splade_doc(
        lambda_q, lambda_d, MaxAggregation.C(), lambda_warmup_steps, hf_id
    )


def splade_encoder_from_pretrained_hf(
    model_id: str,
    maxlen: int = 200,
):
    """Creates a SPLADE DotDense model from a pre-trained HuggingFace MLM checkpoint.

    :param model_id: The HuggingFace model ID for the document encoder
    :param maxlen: Maximum document length
    :returns: (model, init_tasks) tuple
    """
    encoder = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=model_id))
    init_tasks = [HFModelInitFromID.C(model=encoder)]
    tokenizer = HFTokenizerAdapter.C(
        tokenizer=HFTokenizer.C(model_id=model_id),
        converter=TopicTextConverter.C(),
    )

    splade_encoder = SpladeTextEncoder.C(
        aggregation=MaxAggregation.C(),
        encoder=encoder,
        tokenizer=tokenizer,
        maxlen=maxlen,
    )
    return splade_encoder, init_tasks


def splade_from_pretrained_hf(
    model_id: str,
    query_model_id: Optional[str] = None,
    maxlen: int = 200,
    query_maxlen: int = 30,
):
    """Creates a SPLADE DotDense model from a pre-trained HuggingFace MLM checkpoint.

    :param model_id: The HuggingFace model ID for the document encoder
    :param query_model_id: Optional separate model ID for the query encoder
    :param maxlen: Maximum document length
    :param query_maxlen: Maximum query length
    :returns: (model, init_tasks) tuple
    """
    encoder = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=model_id))
    init_tasks = [HFModelInitFromID.C(model=encoder)]
    tokenizer = HFTokenizerAdapter.C(
        tokenizer=HFTokenizer.C(model_id=model_id),
        converter=TopicTextConverter.C(),
    )

    doc_encoder = SpladeTextEncoder.C(
        aggregation=MaxAggregation.C(),
        encoder=encoder,
        tokenizer=tokenizer,
        maxlen=maxlen,
    )

    if query_model_id:
        query_enc = HFMaskedLanguageModel.C(config=HFConfigID.C(hf_id=query_model_id))
        init_tasks.append(HFModelInitFromID.C(model=query_enc))
        query_tok = HFTokenizerAdapter.C(
            tokenizer=HFTokenizer.C(model_id=query_model_id),
            converter=TopicTextConverter.C(),
        )
        query_encoder = SpladeTextEncoder.C(
            aggregation=MaxAggregation.C(),
            encoder=query_enc,
            tokenizer=query_tok,
            maxlen=query_maxlen,
        )
    else:
        query_encoder = SpladeTextEncoder.C(
            aggregation=MaxAggregation.C(),
            encoder=encoder,
            tokenizer=tokenizer,
            maxlen=query_maxlen,
        )

    return DotDense.C(encoder=doc_encoder, query_encoder=query_encoder), init_tasks
