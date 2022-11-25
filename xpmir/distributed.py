from typing import Callable, List
from experimaestro import Config, Param
import torch.nn as nn
import torch
from xpmir.context import Context, InitializationHook
from xpmir.letor import DistributedDeviceInformation
from xpmir.utils import easylog

logger = easylog()


class DistributableModel(Config):
    """A model that can be distributed over GPUs

    Subclasses must implement :py:meth:`distribute_models`
    """

    def distribute_models(self, update: Callable[[nn.Module], nn.Module]):
        """This method is called with an `update` parameter that should be used
        to update all the torch modules that we need to distribute on GPUs"""
        raise NotImplementedError(
            f"distribute_models not implemented in {self.__class__}"
        )


class DistributedHook(InitializationHook):
    """Hook to distribute the model processing

    When in multiprocessing/multidevice, use `torch.nn.parallel.DistributedDataParallel`,
    otherwise use `torch.nn.DataParallel`.
    """

    models: Param[List[DistributableModel]]
    """The model to distribute over GPUs"""

    def after(self, state: Context):
        for model in self.models:
            model.distribute_models(lambda model: self.update(state, model))

    def update(self, state: Context, model: nn.Module) -> nn.Module:
        info = state.device_information
        if isinstance(info, DistributedDeviceInformation):
            logger.info("Using a distributed model with rank=%d", info.rank)
            return nn.parallel.DistributedDataParallel(model, device_ids=[info.rank])
        else:
            n_gpus = torch.cuda.device_count()
            if n_gpus > 1:
                logger.info(
                    "Setting up DataParallel for Splade text encoder (%d GPUs)", n_gpus
                )
                return torch.nn.DataParallel(model)
            else:
                logger.warning("Only one GPU detected, not using data parallel")

        return model