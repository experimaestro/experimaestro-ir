from abc import ABC, abstractmethod
from dataclasses import InitVar
import logging
from typing import Type

from experimaestro import Config, Param

from xpmir.learning import Module
from xpmir.learning.optim import ModuleInitMode, ModuleInitOptions

try:
    from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise


class HFModelConfig(Config, ABC):
    @abstractmethod
    def __call__(self, options: ModuleInitOptions):
        ...


class HFModelConfigId(Config):
    model_id: Param[str]
    """HuggingFace Model ID"""

    def __call__(self, options: ModuleInitOptions, automodel: Type[AutoModel]):
        # Load the model configuration
        config = AutoConfig.from_pretrained(self.model_id)

        if options.mode == ModuleInitMode.NONE or options.mode == ModuleInitMode.RANDOM:
            return config, automodel.from_config(config)

        return config, automodel.from_pretrained(self.model_id, config=config)


class HFModel(Module):
    """Base transformer class from Huggingface

    Loads the pre-trained checkpoint (unless initialized otherwise)
    """

    config: Param[HFModelConfig]
    """Model ID from huggingface"""

    model: InitVar[AutoModel]

    @classmethod
    def from_model_id(cls, model_id: str):
        return cls(config=HFModelConfigId(model_id=model_id))

    @property
    def automodel(self):
        return AutoModel

    def __initialize__(self, options: ModuleInitOptions):
        """Initialize the HuggingFace transformer

        Args:
            options: loader options
        """
        super().__initialize__(options)

        self.config, self.model = self.config(options, self.automodel)


class HFMaskedLanguageModel(HFModel):
    model: InitVar[AutoModelForMaskedLM]

    @property
    def automodel(self):
        return AutoModelForMaskedLM
