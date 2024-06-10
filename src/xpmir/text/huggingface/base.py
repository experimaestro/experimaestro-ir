from abc import ABC, abstractmethod
from dataclasses import InitVar
import logging
import os
from pathlib import Path
from typing import Type
import torch.nn as nn
from experimaestro import Config, Param

from xpmir.learning import Module
from xpmir.learning.optim import ModuleInitMode, ModuleInitOptions
from xpmir.text import TokenizedTexts

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise


class HFModelConfig(Config, ABC):
    """Base class for all HuggingFace model configurations"""

    @abstractmethod
    def __call__(self, options: ModuleInitOptions):
        ...


class HFModelConfigFromId(HFModelConfig):
    model_id: Param[str]
    """HuggingFace Model ID"""

    def get_config(self, options: ModuleInitOptions, automodel: Type[AutoModel]):
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

        # Load the model configuration
        config = AutoConfig.from_pretrained(model_id_or_path)

        # Return it
        return config, model_id_or_path

    def __call__(self, options: ModuleInitOptions, automodel: Type[AutoModel]):
        config, model_id_or_path = self.get_config(options, automodel)

        if options.mode == ModuleInitMode.NONE or options.mode == ModuleInitMode.RANDOM:
            logging.info("Random initialization of HF model")
            return config, automodel.from_config(config)

        logging.info("Loading model from HF (%s)", self.model_id)
        return config, automodel.from_pretrained(model_id_or_path, config=config)


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
        return cls(config=HFModelConfigFromId(model_id=model_id))

    @property
    def automodel(self):
        return AutoModel

    def __initialize__(self, options: ModuleInitOptions):
        """Initialize the HuggingFace transformer

        Args:
            options: loader options
        """
        super().__initialize__(options)

        self.hf_config, self.model = self.config(options, self.automodel)

    @property
    def contextual_model(self) -> nn.Module:
        """Returns the model that only outputs base representations"""

        # This method needs to be updated to cater for various types of models,
        # i.e. MLM, classification, etc.
        return self.model

    def forward(self, tokenized: TokenizedTexts):
        tokenized = tokenized.to(self.model.device)
        return self.model(
            input_ids=tokenized.ids,
            attention_mask=tokenized.mask,
            token_type_ids=tokenized.token_type_ids,
        )


class HFMaskedLanguageModel(HFModel):
    model: InitVar[AutoModelForMaskedLM]

    @property
    def automodel(self):
        return AutoModelForMaskedLM
