"""Utilities for decomposing HuggingFace masked language models.

Decomposes an ``AutoModelForMaskedLM`` into its backbone encoder,
the intermediate MLM transform layers, and the final vocabulary decoder.
"""

import torch.nn as nn
from torch.nn.functional import gelu


class _DistilBertMLMTransform(nn.Module):
    """Wraps DistilBERT's MLM transform layers (vocab_transform + act + LN)."""

    def __init__(self, model):
        super().__init__()
        self.vocab_transform = model.vocab_transform
        self.activation = model.activation
        self.vocab_layer_norm = model.vocab_layer_norm

    def forward(self, x):
        x = self.vocab_transform(x)
        x = self.activation(x)
        x = self.vocab_layer_norm(x)
        return x


class _RobertaMLMTransform(nn.Module):
    """Wraps RoBERTa's LM head transform layers (dense + gelu + LN)."""

    def __init__(self, lm_head):
        super().__init__()
        self.dense = lm_head.dense
        self.layer_norm = lm_head.layer_norm

    def forward(self, x):
        x = self.dense(x)
        x = gelu(x)
        x = self.layer_norm(x)
        return x


def decompose_mlm_model(model):
    """Decompose an AutoModelForMaskedLM into (backbone, transform, decoder).

    The backbone produces ``last_hidden_state``, the transform applies
    the MLM head's intermediate layers, and the decoder is the final
    ``nn.Linear`` projecting to ``vocab_size``.

    Supported architectures: bert, distilbert, roberta, xlm-roberta.

    Returns:
        Tuple[nn.Module, nn.Module, nn.Linear]
    """
    model_type = model.config.model_type
    if model_type == "bert":
        backbone = model.bert
        predictions = model.cls.predictions
        transform = predictions.transform
        decoder = predictions.decoder
    elif model_type == "distilbert":
        backbone = model.distilbert
        transform = _DistilBertMLMTransform(model)
        decoder = model.vocab_projector
    elif model_type in ("roberta", "xlm-roberta"):
        backbone = model.roberta
        transform = _RobertaMLMTransform(model.lm_head)
        decoder = model.lm_head.decoder
    else:
        raise ValueError(
            f"decompose_mlm_model does not support model_type={model_type!r}. "
            "Supported: bert, distilbert, roberta, xlm-roberta."
        )
    return backbone, transform, decoder
