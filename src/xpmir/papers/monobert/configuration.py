from attrs import define, Factory, field
from xpmir.papers.cli import PaperExperiment


@define(kw_only=True)
class Indexation:
    requirements: str = "duration=6 days & cpu(mem=4G, cores=8)"


@define(kw_only=True)
class Learner:
    validation_size: int = 500

    steps_per_epoch: int = 32
    """Number of steps (batches) per epoch"""

    validation_interval: int = field(default=32)
    batch_size: int = 64
    max_epochs: int = 3200
    num_warmup_steps: int = 10000
    warmup_min_factor: float = 0
    lr: float = 3.0e-6
    requirements: str = "duration=4 days & cuda(mem=24G) * 2"

    @validation_interval.validator
    def check_validation_interval(self, attribute, value):
        assert self.max_epochs % value == 0, (
            f"Number of epochs ({self.max_epochs}) is not a multiple "
            f"of validation interval ({value})"
        )


@define(kw_only=True)
class Evaluation:
    requirements: str = "duration=2 days & cuda(mem=24G)"


@define(kw_only=True)
class Retrieval:
    k: int = 1000
    val_k: int = 1000
    batch_size: int = 512


@define(kw_only=True)
class Monobert(PaperExperiment):

    gpu: bool = True
    """Use GPU for computation"""

    indexation: Indexation = Factory(Indexation)
    learner: Learner = Factory(Learner)
    evaluation: Evaluation = Factory(Evaluation)
    retrieval: Retrieval = Factory(Retrieval)
