from typing import List
from functools import cached_property
from xpmir.learning.optim import (
    AdamW,
    Adam,
    SGD,
    ParameterOptimizer,
    RegexParameterFilter,
    get_optimizers,
)
from xpmir.learning.schedulers import LinearWithWarmup
from xpmir.papers import configuration


@configuration
class TransformerOptimization:
    """Configuration for a transformer optimization"""

    scheduler: bool = True
    warmup_min_factor: float = 0

    num_warmup_steps: int = 1000
    batch_size: int = 64
    max_epochs: int = 3200
    steps_per_epoch: int = 32
    """Number of steps (batches) per epoch"""

    optimizer_name: str = "adam-w"
    lr: float = 3.0e-6
    weight_decay: float = 1e-2
    eps: float = 1e-8

    re_no_l2_regularization: List[str] = [r"\.bias$", r"\.LayerNorm\."]
    """Regular expression for layers"""

    def get_optimizer(self, regularization):
        if self.optimizer_name == "adam-w":
            return AdamW(
                lr=self.lr,
                weight_decay=self.weight_decay if regularization else 0,
                eps=self.eps,
            )
        elif self.optimizer_name == "adam-w":
            return Adam(self.lr, weight_decay=0, eps=self.eps)
        elif self.optimizer_name == "sgd":
            return SGD(lr=self.lr)
        else:
            raise ValueError(f"Cannot handle optimizer named {self.optimizer_Name}")

    @cached_property
    def optimizer(self):
        scheduler = (
            LinearWithWarmup(
                num_warmup_steps=self.num_warmup_steps,
                min_factor=self.warmup_min_factor,
            )
            if self.scheduler
            else None
        )

        if not self.re_no_l2_regularization:
            return get_optimizers(
                [
                    ParameterOptimizer(
                        scheduler=scheduler,
                        optimizer=self.get_optimizer(True),
                    ),
                ]
            )

        return get_optimizers(
            [
                ParameterOptimizer(
                    scheduler=scheduler,
                    optimizer=self.get_optimizer(False),
                    filter=RegexParameterFilter(includes=self.re_no_l2_regularization),
                ),
                ParameterOptimizer(
                    scheduler=scheduler,
                    optimizer=self.get_optimizer(True),
                ),
            ]
        )
