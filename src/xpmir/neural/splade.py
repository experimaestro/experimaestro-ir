from typing import List, Optional, Generic
from experimaestro import Config, Param
from datamaestro_text.data.ir import TextItem
import torch.nn as nn
import torch
from xpmir.learning import ModuleInitOptions
from xpmir.distributed import DistributableModel
from xpmir.text.huggingface import (
    OneHotHuggingFaceEncoder,
    TransformerTokensEncoderWithMLMOutput,
)
from xpmir.text import TokenizerOptions
from xpmir.text.huggingface import HFTokenizerBase
from xpmir.text.encoders import (
    TextEncoder,
    TextEncoderBase,
    InputType as EncoderInputType,
    TextsRepresentationOutput,
)
from xpmir.neural.dual import DotDense, ScheduledFlopsRegularizer
from xpmir.text.huggingface.base import HFMaskedLanguageModel
from xpmir.utils.utils import easylog

logger = easylog()


class Aggregation(Config):
    """The aggregation function for Splade"""

    def get_output_module(self, linear: nn.Module) -> nn.Module:
        return AggregationModule(linear, self)


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


class AggregationModule(nn.Module):
    def __init__(self, linear: nn.Linear, aggregation: Aggregation):
        super().__init__()
        self.linear = linear
        self.aggregation = aggregation

    def forward(self, input: torch.Tensor, mask: torch.Tensor):
        return self.aggregation(self.linear(input), mask)


class SpladeTextEncoderModel(nn.Module):
    def __init__(
        self, encoder: TransformerTokensEncoderWithMLMOutput, aggregation: Aggregation
    ):
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

    def __initialize__(self, options: ModuleInitOptions):
        self.encoder.initialize(options)
        self.model = SpladeTextEncoderModel(self.encoder, self.aggregation)

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a batch x vocab tensor"""
        if not isinstance(texts[0], str):
            texts = [text[TextItem].text for text in texts]
        tokenized = self.encoder.batch_tokenize(texts, mask=True, maxlen=self.maxlen)
        out = self.model(tokenized)
        return TextsRepresentationOutput(out, tokenized)

    @property
    def dimension(self):
        return self.encoder.model.config.vocab_size

    def static(self):
        return False

    def distribute_models(self, update):
        self.model = update(self.model)


class IdentityWithBias(nn.Identity):
    def __init__(self):
        # So that set_output_embeddings is happy
        super().__init__()
        self.bias = None


class SpladeTextEncoderV2(
    TextEncoderBase[EncoderInputType, TextsRepresentationOutput],
    DistributableModel,
    Generic[EncoderInputType],
):
    # TODO: use "SpladeTextEncoder" identifier until
    # https://github.com/experimaestro/experimaestro-python/issues/56 is fixed
    __xpmid__ = str(SpladeTextEncoder.__getxpmtype__().identifier)

    """Splade model text encoder (V2)

    It is only a text encoder since the we use `xpmir.neural.dual.DotDense`
    as the scorer class. Compared to V1, it uses the new text HF encoder abstractions.
    """

    tokenizer: Param[HFTokenizerBase[EncoderInputType]]
    """The tokenizer from Hugging Face"""

    encoder: Param[HFMaskedLanguageModel]
    """The encoder from Hugging Face"""

    aggregation: Param[Aggregation]
    """How to aggregate the vectors"""

    maxlen: Param[Optional[int]] = None
    """Max length for texts"""

    def __initialize__(self, options: ModuleInitOptions):
        self.encoder.initialize(options)
        self.tokenizer.initialize(options)

        # Adds the aggregation head right away - this could allows
        # optimization e.g. for the Max aggregation method
        output_embeddings = self.encoder.model.get_output_embeddings()
        assert isinstance(
            output_embeddings, nn.Linear
        ), f"Cannot handle output embeddings of class {output_embeddings.__cls__}"
        self.encoder.model.set_output_embeddings(IdentityWithBias())

        self.aggregation = self.aggregation.get_output_module(output_embeddings)

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

    def distribute_models(self, update):
        self.encoder = update(self.encoder)


def _splade(
    lambda_q: float,
    lambda_d: float,
    aggregation: Aggregation,
    lambda_warmup_steps: int = 0,
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
        lambda_warmup_steps=lambda_warmup_steps,
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
        lambda_warmup_steps=lambda_warmup_steps,
    )


def spladeV1(
    lambda_q: float,
    lambda_d: float,
    lambda_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    """Returns the Splade architecture"""
    return _splade(lambda_q, lambda_d, SumAggregation(), lambda_warmup_steps, hf_id)


def spladeV2_max(
    lambda_q: float,
    lambda_d: float,
    lambda_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    """Returns the Splade-max architecture

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval
    (arXiv:2109.10086)
    """
    return _splade(lambda_q, lambda_d, MaxAggregation(), lambda_warmup_steps, hf_id)


def spladeV2_doc(
    lambda_q: float,
    lambda_d: float,
    lambda_warmup_steps: int = 0,
    hf_id: str = "distilbert-base-uncased",
):
    """Returns the Splade-doc architecture

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval
    (arXiv:2109.10086)
    """
    return _splade_doc(lambda_q, lambda_d, MaxAggregation(), lambda_warmup_steps, hf_id)
