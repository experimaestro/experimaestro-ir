from typing import Callable, List
from experimaestro import Config, Param
import torch.nn as nn
import torch
from xpmir.context import Context, InitializationHook
from xpmir.letor import DistributedDeviceInformation
from xpmir.utils.utils import easylog

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

    When in multiprocessing/multidevice, use `torch.nn.parallel.DistributedDataParallel`
    ,otherwise use `torch.nn.DataParallel`.
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
            if not isinstance(model, nn.DataParallel):
                n_gpus = torch.cuda.device_count()
                if n_gpus > 1:
                    logger.info(
                        "Setting up DataParallel on %d GPUs for model %s",
                        n_gpus,
                        str(model.__class__.__qualname__),
                    )
                    return DataParallel(model)
                else:
                    logger.warning("Only one GPU detected, not using data parallel")

        return model


class DataParallel(torch.nn.DataParallel):
    """This adapter aims to resolve the problem that in the DataParallel,
    it generate the '.module.' in the read and write. Need to rewrite the
    load_state and write_state"""

    module: nn.Module
    """The model to put on multi dataset."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._register_load_state_dict_pre_hook(DataParallel.load_state_dict_hook)

    @staticmethod
    def load_state_dict_hook(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        replaced_keys = [key for key in state_dict if key.startswith(prefix)]
        offset = len(prefix)
        for key in replaced_keys:
            new_key = f"{prefix}module.{key[offset:]}"
            state_dict[new_key] = state_dict.pop(key)

    def state_dict(self, **kwargs):
        return self.module.state_dict(**kwargs)
