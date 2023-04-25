import numpy as np
from experimaestro import Config, Param
from functools import cached_property


class Random(Config):
    """Random configuration"""

    seed: Param[int] = 0
    """The seed to use so the random process is deterministic"""

    @cached_property
    def state(self) -> np.random.RandomState:
        return np.random.RandomState(self.seed)

    def __getstate__(self):
        return {"seed": self.seed}
