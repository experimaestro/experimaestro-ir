from typing import Any, Callable, Iterator, Optional, Tuple
from experimaestro import Config, Param
from .schedulers import Scheduler
import torch


class Optimizer(Config):
    def __call__(self, parameters) -> torch.optim.Optimizer:
        raise NotImplementedError()


class Adam(Optimizer):
    lr: Param[float] = 1e-3

    def __call__(self, parameters):
        from torch.optim import Adam

        return Adam(parameters, lr=self.lr)


class AdamW(Optimizer):
    """Adam optimizer that takes into account the regularization"""

    lr: Param[float] = 1e-3
    weight_decay: Param[float] = 1e-2

    def __call__(self, parameters):
        from torch.optim import AdamW

        return AdamW(parameters, lr=self.lr, weight_decay=self.weight_decay)


class Module(Config, torch.nn.Module):
    """A module contains parameters"""

    pass


class ParameterFilter(Config):
    def __call__(self, name, params) -> bool:
        return True


class ParameterOptimizer(Config):
    """Associates an optimizer with a list of parameters to optimize"""

    optimizer: Param[Optimizer]
    """The optimizer"""

    scheduler: Param[Scheduler]
    """The optional scheduler"""

    module: Param[Optional[Module]]
    """The module from which parameters should be extracted"""

    filter: Param[Optional[ParameterFilter]] = ParameterFilter()
    """How parameters should be selected for this (by default, use them all)"""

    def create_optimizer(
        self, module: Module, filter: Callable[[str, Any], bool]
    ) -> torch.optim.Optimizer:
        """Returns a (pytorch) optimizer"""
        module = self.module or module
        optimizer = self.optimizer(
            param
            for name, param in module.named_parameters()
            if (self.filter is None or self.filter(name, param)) and filter(name, param)
        )
        return optimizer
