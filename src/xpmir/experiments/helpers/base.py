from experimaestro.experiments import configuration, ConfigurationBase
from experimaestro.launcherfinder import find_launcher

from functools import cached_property as attrs_cached_property
from xpmir.learning.devices import CudaDevice, BestDevice, Device
from xpmir.letor import Random


PaperExperiment = ConfigurationBase


@configuration()
class NeuralIRExperiment(ConfigurationBase):
    """Settings most neural IR experiments"""

    gpu: bool = True
    """Use GPU for computation"""

    use_best_device: bool = False
    """Use best GPU device"""

    seed: int = 0
    """The seed used for experiments"""

    @attrs_cached_property
    def random(self):
        return Random(seed=self.seed)

    @attrs_cached_property
    def device(self) -> Device:
        if self.use_best_device:
            return BestDevice()
        return CudaDevice() if self.gpu else Device()


@configuration()
class LauncherSpecification:
    """Launcher specification

    This allows requesting computational resources such as 2 GPUs with more than
    12Go of memory)
    """

    requirements: str

    @attrs_cached_property
    def launcher(self):
        return find_launcher(self.requirements)
