from torch.optim.lr_scheduler import LambdaLR
from experimaestro import Config, Param


class Scheduler(Config):
    def __call__(self, optimizer, *, last_epoch=-1, **kwargs):
        raise NotImplementedError(f"Not implemented in {self.__class__}")


class LinearWithWarmup(Scheduler):
    """Linear warmup followed by decay

    Attributes:
        num_warmup_steps: Nummber of warmup steps
        num_training_steps: Total number of training steps
    """

    num_warmup_steps: Param[int]
    num_training_steps: Param[int]

    def lr_lambda(self, current_step: int):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.num_warmup_steps)),
        )

    def __call__(self, optimizer, *, last_epoch=-1):
        return LambdaLR(optimizer, self.lr_lambda)
