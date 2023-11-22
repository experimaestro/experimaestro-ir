from experimaestro.launcherfinder import find_launcher
from omegaconf import MISSING
from xpmir.letor import Random

from xpmir.learning.devices import Device, CudaDevice
from xpmir.papers import configuration, attrs_cached_property


@configuration()
class PaperExperiment:
    id: str = MISSING
    """The experiment ID

    The ID should be unique, and might be used when exporting the model e.g. to
    HuggingFace
    """

    title: str = ""
    """The model title

    The title is used to generate report
    """

    description: str = ""
    """A description of the model

    This description is used to generate reports
    """


@configuration()
class NeuralIRExperiment(PaperExperiment):
    """Settings most neural IR experiments"""

    gpu: bool = True
    """Use GPU for computation"""

    seed: int = 0
    """The seed used for experiments"""

    @attrs_cached_property
    def random(self):
        return Random(seed=self.seed)

    @attrs_cached_property
    def device(self) -> Device:
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
