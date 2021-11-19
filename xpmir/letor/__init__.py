import torch
from experimaestro import Config, Param
from experimaestro.compat import cached_property
import numpy as np


class Random(Config):
    seed: Param[int] = 0

    @cached_property
    def state(self) -> np.random.RandomState:
        return np.random.RandomState(self.seed)

    def __getstate__(self):
        return {"seed": self.seed}


class Device(Config):
    """Device to use

    Attributes:

        gpu: use CUDA if available
        gpu_determ: Use deterministic CUDA (CuDNN)
    """

    gpu: Param[bool] = False
    gpu_determ: Param[bool] = False

    def __call__(self, logger):
        """Called by experimaestro to substitute object at run time"""
        device = torch.device("cpu")

        if self.gpu:
            if not torch.cuda.is_available():
                logger.error(
                    "gpu=True, but CUDA is not available. Falling back on CPU."
                )
            else:
                if self.gpu_determ:
                    if logger is not None:
                        logger.debug("using GPU (deterministic)")
                else:
                    if logger is not None:
                        logger.debug("using GPU (non-deterministic)")
                device = torch.device("cuda")
                torch.backends.cudnn.deterministic = self.gpu_determ
        return device


# Default device is the CPU
DEFAULT_DEVICE = Device()
