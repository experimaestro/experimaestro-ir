import os
from abc import ABC
from contextlib import nullcontext
from dataclasses import InitVar
from pathlib import Path
from typing import Generic, Optional, Type, TypeVar

import torch.nn as nn
from experimaestro import field, Config, Param, LightweightTask

from xpm_torch import Module
from xpm_torch.configuration import FabricConfiguration
from xpmir.text import TokenizedTexts
from functools import lru_cache

import logging

logger = logging.getLogger(__name__)


try:
    from transformers import (
        AutoConfig,
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
    )
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise


@lru_cache
def is_local_files_only():
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ["1", "true", "on"]


def _resolve_model_path(model_id: str, automodel: Type[AutoModel]):
    """Resolves the model ID or local path, checking XPMIR_TRANSFORMERS_CACHE"""
    model_id_or_path = model_id

    if model_path := os.environ.get("XPMIR_TRANSFORMERS_CACHE", None):
        path = (
            Path(model_path)
            / Path(f"{automodel.__module__}.{automodel.__qualname__}")
            / Path(model_id)
        )
        if path.is_dir():
            logging.warning("Using saved model from %s", path)
            model_id_or_path = path
        else:
            logging.warning("Could not find saved model in %s, using HF loading", path)

    return model_id_or_path


class HFConfig(Config):
    """Base configuration for HuggingFace models"""

    pass


class HFConfigID(HFConfig):
    """Configuration identified by a HuggingFace model ID"""

    hf_id: Param[str]
    """HuggingFace model ID (e.g. ``distilbert-base-uncased``)"""


ConfigT = TypeVar("ConfigT", bound=HFConfig)


class HFModel(Module, Generic[ConfigT]):
    """Base transformer class from Huggingface

    Model structure is created during ``__initialize__`` from the
    :attr:`config` when available.  Pretrained weights can be loaded
    via init tasks such as :class:`HFModelInitFromID` or
    :class:`HFFromCheckpoint`.
    """

    config: Param[ConfigT]
    """HuggingFace model configuration"""

    model: InitVar[AutoModel]
    """The HF model"""

    @property
    def autoconfig(self):
        return AutoConfig

    @property
    def automodel(self):
        return AutoModel

    def __initialize__(self):
        """Creates the model structure from config.hf_id (no pretrained weights)"""
        if isinstance(self.config, HFConfigID):
            hf_id = self.config.hf_id
            model_id_or_path = _resolve_model_path(hf_id, self.automodel)
            hf_config = self.autoconfig.from_pretrained(
                model_id_or_path,
                trust_remote_code=True,
                local_files_only=is_local_files_only(),
            )
            self.hf_config = hf_config
            logging.info(
                "Creating model structure from config (%s) with %s.%s",
                hf_id,
                self.automodel.__module__,
                self.automodel.__name__,
            )
            self.model = self.automodel.from_config(hf_config)

    @property
    def contextual_model(self) -> nn.Module:
        """Returns the model that only outputs base representations"""
        return self.model

    def forward(self, tokenized: TokenizedTexts):
        tokenized = tokenized.to(self.model.device)
        kwargs = {}
        if tokenized.token_type_ids is not None:
            kwargs["token_type_ids"] = tokenized.token_type_ids

        return self.model(
            input_ids=tokenized.ids,
            attention_mask=tokenized.mask,
        )


class HFMaskedLanguageModel(HFModel):
    model: InitVar[AutoModelForMaskedLM]

    @property
    def automodel(self):
        return AutoModelForMaskedLM

    def decompose(self):
        """Decompose into (backbone, transform, decoder).

        See :func:`~xpmir.text.huggingface.decompose.decompose_mlm_model`
        for details.
        """
        from xpmir.text.huggingface.decompose import decompose_mlm_model

        return decompose_mlm_model(self.model)


class HFSequenceClassification(HFModel):
    """HuggingFace model for sequence classification"""

    model: InitVar[AutoModelForSequenceClassification]

    n_labels: Param[int] = field(default=1, ignore_default=True)

    # override
    def __initialize__(self):
        """Creates the model structure from config.hf_id (no pretrained weights)
        Checks and modifies the config to match "n_labels"
        """
        if isinstance(self.config, HFConfigID):
            hf_id = self.config.hf_id
            model_id_or_path = _resolve_model_path(hf_id, self.automodel)
            hf_config = self.autoconfig.from_pretrained(
                model_id_or_path,
                trust_remote_code=True,
                local_files_only=is_local_files_only(),
            )

            # ensure that num_labels is one for a Cross-encoder
            if hasattr(hf_config, "num_labels"):
                if hf_config.num_labels != self.n_labels:
                    logger.debug(
                        f"hf config 'n_labels' was {hf_config.num_labels}, setting it to {self.n_labels}"
                    )
                hf_config.num_labels = self.n_labels

            else:
                self.logger.warning(
                    "no 'num_labels param found in config, check that classifier outputs one label"
                )

            self.hf_config = hf_config
            logger.info(
                "Creating model structure from config (%s) with %s.%s",
                hf_id,
                self.automodel.__module__,
                self.automodel.__name__,
            )
            self.model = self.automodel.from_config(hf_config)

    @property
    def automodel(self):
        return AutoModelForSequenceClassification


class HFModelInitBase(LightweightTask, ABC):
    """Base class for initializing HF models"""

    model: Param[HFModel[HFConfigID]]

    def __validate__(self):
        assert isinstance(self.model.config, HFConfigID), (
            f"model.config must be an HFConfigID, got {type(self.model.config)}"
        )

    fabric: Param[Optional[FabricConfiguration]]
    """The fabric configuration to use for initialization. When set, model
    creation runs inside ``fabric.init_module()`` so that parameters are
    allocated directly on the target device and dtype.
    See https://lightning.ai/docs/fabric/stable/advanced/model_init.html
    """

    def _init_context(self, empty_init: bool):
        """Returns a context manager for model initialization.

        When ``self.fabric`` is set, returns ``fabric.init_module(empty_init)``;
        otherwise returns a no-op context.

        :param empty_init: If True, parameters are created on the meta device
            (no memory allocated). Use True when loading weights from a
            checkpoint (pretrained / saved), False when random init is needed.
        """
        if self.fabric is not None:
            return self.fabric.get_fabric().init_module(empty_init=empty_init)
        return nullcontext()


class HFModelInitFromID(HFModelInitBase):
    """Load pretrained weights from a HuggingFace Hub model ID.

    Uses ``model.config.hf_id`` to resolve the model.
    """

    def execute(self):
        hf_id = self.model.config.hf_id
        model_id_or_path = _resolve_model_path(hf_id, self.model.automodel)
        config = self.model.autoconfig.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            local_files_only=is_local_files_only(),
        )
        self.model.hf_config = config
        logging.info(
            "Loading pretrained model from HF (%s) with %s.%s",
            hf_id,
            self.model.automodel.__module__,
            self.model.automodel.__name__,
        )
        with self._init_context(empty_init=True):
            self.model.model = self.model.automodel.from_pretrained(
                model_id_or_path,
                config=config,
                trust_remote_code=True,
                local_files_only=is_local_files_only(),
            )
        self.model._initialized = True


class HFFromCheckpoint(HFModelInitBase):
    """Load from a local checkpoint.

    Uses ``model.config.hf_id`` for the architecture config, then loads weights
    from ``checkpoint``.
    """

    checkpoint: Param[Path]
    """The checkpoint path to load weights from"""

    def execute(self):
        hf_id = self.model.config.hf_id
        model_id_or_path = _resolve_model_path(hf_id, self.model.automodel)
        config = self.model.autoconfig.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            local_files_only=is_local_files_only(),
        )
        self.model.hf_config = config
        logging.info(
            "Loading model from checkpoint %s (config from %s)",
            self.checkpoint,
            hf_id,
        )
        with self._init_context(empty_init=True):
            self.model.model = self.model.automodel.from_pretrained(
                self.checkpoint,
                config=config,
                trust_remote_code=True,
                local_files_only=is_local_files_only(),
            )
        self.model._initialized = True
