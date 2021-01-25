from tqdm import tqdm
import torch
from experimaestro import param, option, config, pathoption

# from onir import util, trainers
from xpmir.interfaces import apex

# from onir.rankers import Ranker
# from onir.vocab import Vocab
# from onir.log import Logger


from experimaestro import config, Param
from cached_property import cached_property
import numpy as np


@config()
class Random:
    seed: Param[int] = 0

    @cached_property
    def state(self):
        return np.random.RandomState(self.seed)

    def __getstate__(self):
        return {"seed": self.seed}


@config()
class Device:
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
