from experimaestro import Config, Param
from experimaestro.compat import cached_property
import numpy as np
from xpmir.utils.utils import easylog

# flake8: noqa: F401
from .devices import (
    DEFAULT_DEVICE,
    Device,
    DeviceInformation,
    DistributedDeviceInformation,
)

logger = easylog()


class Random(Config):
    """Random configuration"""

    seed: Param[int] = 0
    """The seed to use so the random process is deterministic"""

    @cached_property
    def state(self) -> np.random.RandomState:
        return np.random.RandomState(self.seed)

    def __getstate__(self):
        return {"seed": self.seed}
