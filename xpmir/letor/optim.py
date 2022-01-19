from typing import Any, Callable, Iterator, List, Optional, Tuple, TYPE_CHECKING
from experimaestro import Config, Param
import torch
from .schedulers import Scheduler
from xpmir.utils import easylog
from xpmir.letor.metrics import ScalarMetric

if TYPE_CHECKING:
    from xpmir.letor.context import TrainerContext

logger = easylog()


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

    def __call__(self, *args, **kwargs):
        return torch.nn.Module.__call__(self, *args, **kwargs)


class ParameterFilter(Config):
    def __call__(self, name, params) -> bool:
        return True


class ParameterOptimizer(Config):
    """Associates an optimizer with a list of parameters to optimize"""

    optimizer: Param[Optimizer]
    """The optimizer"""

    scheduler: Param[Optional[Scheduler]]
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


class DuplicateParameterFilter:
    """Filters out already optimized parameters"""

    def __init__(self):
        self.parameters = set()

    def __call__(self, name, params):
        if params in self.parameters:
            return False
        self.parameters.add(params)
        return True


class ScheduledOptimizer:
    def initialize(
        self,
        param_optimizers: List[ParameterOptimizer],
        num_training_steps: int,
        module: Module,
        use_scaler: bool,
    ):
        self.schedulers = []
        self.scheduler_factories = []
        self.optimizers = []
        self.scheduler_steps = -1  # Number of scheduler steps
        self.num_training_steps = num_training_steps

        filter = DuplicateParameterFilter()
        for param_optimizer in param_optimizers:
            optimizer = param_optimizer.create_optimizer(module, filter)
            self.optimizers.append(optimizer)
            self.scheduler_factories.append(param_optimizer.scheduler)

        self.reset_schedulers()

        assert len(self.schedulers) == len(self.optimizers)

        if use_scaler:
            logger.info("Using GradScaler when optimizing")
        self.scaler = torch.cuda.amp.GradScaler() if use_scaler else None

    def load_state_dict(self, state):
        for optimizer, optimizer_state in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(optimizer_state)

        if self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])

        # Re-create schedulers
        self.scheduler_steps = state["scheduler_steps"]
        self.reset_schedulers()

    def reset_schedulers(self):
        self.schedulers = []
        for optimizer, scheduler_factory in zip(
            self.optimizers, self.scheduler_factories
        ):
            if scheduler_factory is None:
                self.schedulers.append(None)
            else:
                self.schedulers.append(
                    scheduler_factory(
                        optimizer,
                        self.num_training_steps,
                        last_epoch=self.scheduler_steps,
                    )
                )

    def state_dict(self):
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "scheduler_steps": self.scheduler_steps,
        }

    def scale(self, loss: torch.Tensor):
        if self.scaler is None:
            return loss
        return self.scaler.scale(loss)

    def zero_grad(self):
        """Zero-grad for all optimizers"""
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def optimizer_step(self, context: "TrainerContext"):
        """Performs an optimizer step (using the scaler if defined)"""
        if self.scaler is None:
            for optimizer in self.optimizers:
                optimizer.step()

        else:
            for optimizer in self.optimizers:
                self.scaler.step(optimizer)
            context.add_metric(
                ScalarMetric("gradient/scaler", self.scaler.get_scale(), 1)
            )
            self.scaler.update()

    def scheduler_step(self, context: "TrainerContext"):
        """Performs a step for all the schedulers"""
        for ix, scheduler in enumerate(self.schedulers):
            if scheduler is not None:
                for p_ix, lr in enumerate(scheduler.get_last_lr()):
                    context.add_metric(
                        ScalarMetric(f"gradient/scheduler/{ix+1}/{p_ix+1}", lr, 1)
                    )
                scheduler.step()
        self.scheduler_steps += 1
