from experimaestro.experiments import configuration, ConfigurationBase

from functools import cached_property as attrs_cached_property
from xpm_torch import Random
from .launchers import LauncherSpecification  # noqa

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
