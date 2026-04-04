from pathlib import Path
from typing import Optional, Generic
from experimaestro import field, Config, Param
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
from xpmir.neural.dual import DotDense, DualModuleLoader, ScheduledFlopsRegularizer
from xpmir.neural.sentence_transformers import SpladeLoaderMixin
from xpmir.text.huggingface.base import (
    HFConfigID,
    HFMaskedLanguageModel,
    HFModelInitFromID,
)
from xpm_torch.module import initialized
import logging

logger = logging.getLogger(__name__)

# Sparton: GPU-accelerated SPLADE kernels
# https://github.com/thongnt99/sparton — Apache License 2.0
try:
    from xpmir.neural._sparton import SpartonHead

    _SPARTON_AVAILABLE = True
except ImportError:
    _SPARTON_AVAILABLE = False


class Aggregation(Config):
    """The aggregation function for SPLADE"""

    def get_output_module(self, transform: nn.Module, decoder: nn.Linear) -> nn.Module:
        """Returns an nn.Module: (hidden_states, mask) -> [B, V] sparse reps"""
        raise NotImplementedError


class MaxAggregation(Aggregation):
    """Aggregate using a max"""

    def get_output_module(self, transform, decoder):
        if _SPARTON_AVAILABLE:
            logger.info("Using Sparton fused Triton kernel for MaxAggregation")
            return SpartonAggregationModule(transform, decoder)
        logger.info("Using PyTorch fallback for MaxAggregation (Sparton unavailable)")
        return PyTorchAggregationModule(transform, decoder, self)

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

    def get_output_module(self, transform, decoder):
        return PyTorchAggregationModule(transform, decoder, self)

    def __call__(self, logits, mask):
        assert logits.shape[:2] == mask.shape[:2], (
            f"Shape mismatch: logits {logits.shape} vs mask {mask.shape}"
        )
        return torch.sum(
            torch.log1p(torch.relu(logits) * mask.to(logits.device).unsqueeze(-1)),
            dim=1,
        )


class PyTorchAggregationModule(nn.Module):
    """PyTorch fallback: transform -> decoder -> aggregation"""

    def __init__(
        self, transform: nn.Module, decoder: nn.Linear, aggregation: Aggregation
    ):
        super().__init__()
        self.transform = transform
        self.decoder = decoder
        self.aggregation = aggregation

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        logits = self.decoder(self.transform(hidden_states))
        return self.aggregation(logits, mask)


class SpartonAggregationModule(nn.Module):
    """Fused Triton kernel: transform -> sparton (matmul+max+relu+log1p).

    Falls back to PyTorch on CPU.
    """

    def __init__(self, transform: nn.Module, decoder: nn.Linear):
        super().__init__()
        self.transform = transform
        self.decoder = decoder
        self._sparton_head = SpartonHead(
            decoder.out_features,
            decoder.in_features,
            use_bias=(decoder.bias is not None),
        )
        self._sparton_head.tie_weights(decoder)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        hidden_states = self.transform(hidden_states)
        if hidden_states.is_cuda:
            return self._sparton_head(hidden_states, mask)
        # CPU fallback: standard PyTorch max+relu+log1p
        logits = self.decoder(hidden_states)
        values, _ = torch.max(
            torch.relu(logits) * mask.to(logits.device).unsqueeze(-1), dim=1
        )
        return torch.log1p(values.clamp(min=0))


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

    maxlen: Param[Optional[int]] = field(default=None, ignore_default=True)
    """Max length for texts"""

    def __initialize__(self):
        """Module initialization: initializes the encoder and tokenizer, and
        decomposes the MLM model into backbone + aggregation head."""

        self.encoder.initialize()
        self.tokenizer.initialize()

        backbone, transform, decoder = self.encoder.decompose()
        self._backbone = backbone
        self._head = self.aggregation.get_output_module(transform, decoder)

    @initialized
    def forward(self, texts: EncoderInputType) -> TextsRepresentationOutput:
        """Returns a batch x vocab tensor"""
        tokenized = self.tokenizer.tokenize(
            texts, options=TokenizerOptions(self.maxlen)
        )
        tokenized = tokenized.to(self.encoder.model.device)

        kwargs = {}
        if tokenized.token_type_ids is not None:
            kwargs["token_type_ids"] = tokenized.token_type_ids

        hidden = self._backbone(
            input_ids=tokenized.ids,
            attention_mask=tokenized.mask,
            **kwargs,
        ).last_hidden_state

        value = self._head(hidden, tokenized.mask)
        return TextsRepresentationOutput(value, tokenized)

    def save_model(self, path: Path):
        """Save the HF model in standard pretrained format.

        With tie_weights, the decoder linear remains in the HF model's module
        tree, so ``save_pretrained`` saves all weights naturally.
        """
        path.mkdir(parents=True, exist_ok=True)
        self.encoder.model.save_pretrained(path)
        self.tokenizer.tokenizer.tokenizer.save_pretrained(path)

    def load_model(self, path: Path):
        """Load from HF pretrained format, or fall back to safetensors/pth."""
        config_path = path / "config.json"
        if config_path.exists():
            from transformers import AutoModelForMaskedLM

            self.encoder.model = AutoModelForMaskedLM.from_pretrained(path)
            backbone, transform, decoder = self.encoder.decompose()
            self._backbone = backbone
            self._head = self.aggregation.get_output_module(transform, decoder)
        else:
            super().load_model(path)

    @property
    def dimension(self):
        return self.encoder.model.config.vocab_size

    def static(self):
        return False


class SpladeModuleLoader(SpladeLoaderMixin, DualModuleLoader):
    """ModuleLoader for SPLADE models with separate encoder DataPaths.

    Has distinct ``encoder_path`` and ``query_encoder_path`` DataPaths so
    each encoder is serialized independently. Overrides
    ``__xpm_serialize__`` to map field names to ST-compatible directory
    names (``document_0_MLMTransformer``, ``query_0_MLMTransformer``).

    Inherits :class:`~xpmir.neural.sentence_transformers.SpladeLoaderMixin`
    for ST config writing and README sections.
    """

    # ST-compatible directory name mapping
    _ST_NAMES = {
        "encoder_path": "document_0_MLMTransformer",
        "query_encoder_path": "query_0_MLMTransformer",
    }

    def __xpm_serialize__(self, context):
        result = {}
        for argument, value in self.__xpm__.xpmvalues():
            if argument.is_data and value is not None:
                st_name = self._ST_NAMES.get(argument.name, argument.name)
                result[argument.name] = context.serialize(
                    context.var_path + [st_name], Path(value), self
                )
        return result


class SpladeScorer(DotDense):
    """DotDense subclass for SPLADE models.

    Overrides :meth:`loader_config` to return :class:`SpladeModuleLoader`,
    which has separate DataPaths per encoder and writes ST SparseEncoder
    config files when exporting to HuggingFace Hub.
    """

    CONFIG_LOADER = SpladeModuleLoader.C

    def _has_separate_query_model(self) -> bool:
        """Check if query and doc encoders use different HF models."""
        if self.query_encoder is None or self.query_encoder is self.encoder:
            return False
        # Both are SpladeTextEncoder — check if they share the HF model
        if isinstance(self.encoder, SpladeTextEncoder) and isinstance(
            self._query_encoder, SpladeTextEncoder
        ):
            return self.encoder.encoder is not self._query_encoder.encoder
        return True


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
        SpladeScorer.C(encoder=doc_encoder, query_encoder=query_encoder),
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
        SpladeScorer.C(encoder=doc_encoder, query_encoder=query_encoder),
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
