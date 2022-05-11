from torch.optim.lr_scheduler import LambdaLR
from experimaestro import Config, Param


class Scheduler(Config):
    """Base class for all optimizers schedulers"""

    def __call__(self, optimizer, num_training_steps: int, *, last_epoch=-1, **kwargs):
        raise NotImplementedError(f"Not implemented in {self.__class__}")


class LinearWithWarmup(Scheduler):
    """Linear warmup followed by decay"""

    num_warmup_steps: Param[int]
    """Number of warmup steps"""

    min_factor: Param[float] = 0.0
    """Minimum multiplicative factor"""

    def lr_lambda(self, current_step: int, num_training_steps: int):
        # Still warming up
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))

        # Not warming up: the ratio is between 1 (after warmup) and 0 (at the end)
        factor = max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - self.num_warmup_steps)),
        )

        # Shift/scale so it is between 1 and min factor
        return (factor + self.min_factor) / (1.0 + self.min_factor)

    def __call__(self, optimizer, num_training_steps, *, last_epoch=-1):
        return LambdaLR(
            optimizer,
            lambda current_step: self.lr_lambda(current_step, num_training_steps),
            last_epoch=last_epoch,
        )


class CosineWithWarmup(Scheduler):
    """Cosine schedule with warmup

    Uses the implementation of the transformer library

    https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.get_cosine_schedule_with_warmup

    """

    num_warmup_steps: Param[int]
    """Number of warmup steps"""

    num_cycles: Param[float] = 0.5
    """Number of cycles"""

    def __call__(self, optimizer, num_training_steps, *, last_epoch=-1):
        import transformers

        return transformers.get_cosine_schedule_with_warmup(
            optimizer,
            self.num_warmup_steps,
            num_training_steps,
            self.num_cycles,
            last_epoch=last_epoch,
        )
