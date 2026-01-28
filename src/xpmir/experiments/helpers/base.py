from experimaestro.experiments import configuration, ConfigurationBase
from experimaestro.launcherfinder import find_launcher

from functools import cached_property as attrs_cached_property
from xpm_torch import Random


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
        return Random.C(seed=self.seed)


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
