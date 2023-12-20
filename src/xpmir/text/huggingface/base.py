import logging

from experimaestro import Param

from xpmir.learning import Module
from xpmir.learning.optim import ModuleInitMode, ModuleInitOptions

try:
    from transformers import AutoConfig, AutoModel
except Exception:
    logging.error("Install huggingface transformers to use these configurations")
    raise


class HFBaseModel(Module):
    """Base class for HuggingFace models"""

    pass


class HFNamedModel(Module):
    """Base transformer class from Huggingface

    Loads the pre-trained checkpoint (unless initialized otherwise)
    """

    model_id: Param[str]
    """Model ID from huggingface"""

    @property
    def automodel(self):
        return AutoModel

    def __initialize__(self, options: ModuleInitOptions):
        """Initialize the HuggingFace transformer

        Args:
            options: loader options
        """
        super().__initialize__(options)

        # Load the model configuration
        config = AutoConfig.from_pretrained(self.model_id)

        if options.mode == ModuleInitMode.NONE or options.mode == ModuleInitMode.RANDOM:
            self.model = self.automodel.from_config(config)
        else:
            self.model = self.automodel.from_pretrained(self.model_id, config=config)
