from functools import cached_property
from xpmir.letor.optim import (
    AdamW,
    ParameterOptimizer,
    RegexParameterFilter,
    get_optimizers,
)
from xpmir.letor.schedulers import LinearWithWarmup
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

    lr: float = 3.0e-6
    weight_decay: float = 1e-2

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

        return get_optimizers(
            [
                ParameterOptimizer(
                    scheduler=scheduler,
                    optimizer=AdamW(lr=self.lr, weight_decay=0, eps=self.lr),
                    filter=RegexParameterFilter(
                        includes=[r"\.bias$", r"\.LayerNorm\."]
                    ),
                ),
                ParameterOptimizer(
                    scheduler=scheduler,
                    optimizer=AdamW(
                        lr=self.lr, weight_decay=self.weight_decay, eps=self.lr
                    ),
                ),
            ]
        )
