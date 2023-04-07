from attrs import Factory, field
from experimaestro.launcherfinder import find_launcher
from xpmir.papers import configuration
from xpmir.papers.helpers import LauncherSpecification
from xpmir.papers.helpers.optim import TransformerOptimization
from xpmir.papers.helpers.msmarco import RerankerMSMarcoV1Configuration


@configuration()
class Indexation(LauncherSpecification):
    requirements: str = "duration=6 days & cpu(mem=4G, cores=8)"


@configuration()
class Learner:
    validation_interval: int = field(default=32)
    validation_top_k: int = 1000

    optimization: TransformerOptimization = Factory(TransformerOptimization)
    requirements: str = "duration=4 days & cuda(mem=24G) * 2"

    # FIXME: still not good!
    # def __attrs_post_init__(self):
    #     assert self.optimizer.max_epochs % self.validation_interval == 0, (
    #         f"Number of epochs ({self.optimizer.max_epochs}) is not a multiple "
    #         f"of validation interval ({self.validation_interval})"
    #     )


@configuration()
class Retrieval:
    k: int = 1000
    batch_size: int = 512
    requirements: str = "duration=2 days & cuda(mem=24G)"


@configuration()
class Monobert(RerankerMSMarcoV1Configuration):
    indexation: Indexation = Factory(Indexation)
    retrieval: Retrieval = Factory(Retrieval)

    monobert: Learner = Factory(Learner)
