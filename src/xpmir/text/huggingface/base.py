import logging
import os
from abc import ABC, abstractmethod
from dataclasses import InitVar
from pathlib import Path
from typing import Any, Tuple, Type

import torch.nn as nn
from experimaestro import Config, Param, LightweightTask

from xpm_torch import Module
from xpmir.text import TokenizedTexts
from functools import lru_cache


try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise


@lru_cache
def is_local_files_only():
    return os.environ.get("HF_HUB_OFFLINE", "").lower() in ["1", "true", "on"]


class HFModelConfig(Config, ABC):
    """Base class for all HuggingFace model configurations"""

    @abstractmethod
    def __call__(
        self,
        autoconfig: Type[AutoConfig],
        automodel: Type[AutoModel],
    ) -> Tuple[Any, Any]:
        """Returns a configuration and a model

        By default creates structure only (from_config). Call
        :meth:`use_pretrained` to switch to loading pretrained weights.

        :param autoconfig: configuration factory
        :param automodel: model factory
        :return: a Tuple (configuration, model)
        """
        ...

    @abstractmethod
    def use_pretrained(self):
        """Switch this config to load pretrained weights on next initialize"""
        ...


class HFModelConfigFromId(HFModelConfig):
    model_id: Param[str]
    """HuggingFace Model ID"""

    def __post_init__(self):
        self._create_model = self._from_config

    def _resolve_model_path(self, automodel: Type[AutoModel]):
        """Resolves the model ID or local path"""
        model_id_or_path = self.model_id

        # Use saved models
        if model_path := os.environ.get("XPMIR_TRANSFORMERS_CACHE", None):
            path = (
                Path(model_path)
                / Path(f"{automodel.__module__}.{automodel.__qualname__}")
                / Path(self.model_id)
            )
            if path.is_dir():
                logging.warning("Using saved model from %s", path)
                model_id_or_path = path
            else:
                logging.warning(
                    "Could not find saved model in %s, using HF loading", path
                )

        return model_id_or_path

    def _from_config(self, config, model_id_or_path, automodel):
        logging.info("Structure-only initialization of HF model")
        return automodel.from_config(config, trust_remote_code=True)

    def _from_pretrained(self, config, model_id_or_path, automodel):
        logging.info(
            "Loading pretrained model from HF (%s) with %s.%s",
            self.model_id,
            automodel.__module__,
            automodel.__name__,
        )
        return automodel.from_pretrained(
            model_id_or_path,
            config=config,
            trust_remote_code=True,
            local_files_only=is_local_files_only(),
        )

    def __call__(
        self,
        autoconfig: Type[AutoConfig],
        automodel: Type[AutoModel],
    ):
        model_id_or_path = self._resolve_model_path(automodel)
        config = autoconfig.from_pretrained(
            model_id_or_path,
            trust_remote_code=True,
            local_files_only=is_local_files_only(),
        )
        return config, self._create_model(config, model_id_or_path, automodel)

    def use_pretrained(self):
        if self._create_model == self._from_pretrained:
            logging.warning(
                "HFModelConfigFromId(%s): use_pretrained called more than once",
                self.model_id,
            )
        self._create_model = self._from_pretrained


class HFModel(Module):
    """Base transformer class from Huggingface

    The config specifies the architecture
    """

    config: Param[HFModelConfig]
    """Model ID from huggingface"""

    model: InitVar[AutoModel]
    """The HF model"""

    @classmethod
    def from_pretrained_id(cls, model_id: str):
        return cls.C(config=HFModelConfigFromId.C(model_id=model_id))

    @property
    def autoconfig(self):
        return AutoConfig

    @property
    def automodel(self):
        return AutoModel

    def __initialize__(self):
        """Initialize the HuggingFace transformer (structure only)"""
        super().__initialize__()

        self.hf_config, self.model = self.config(
            self.autoconfig, self.automodel
        )

    @property
    def contextual_model(self) -> nn.Module:
        """Returns the model that only outputs base representations"""

        # This method needs to be updated to cater for various types of models,
        # i.e. MLM, classification, etc.
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


class LoadFromHFCheckpoint(LightweightTask):
    """Switches an HFModel's config to load pretrained weights

    This runs as an init_task before the main task's execute(). It switches
    the config so that when initialize() is later called (e.g. by the Learner),
    it uses from_pretrained instead of from_config.
    """

    model: Param[HFModel]

    def execute(self):
        self.model.config.use_pretrained()
