from attrs import define, Factory, field
from xpmir.papers.pipelines.msmarco import RerankerMSMarcoV1Configuration


@define(kw_only=True)
class Indexation:
    requirements: str = "duration=6 days & cpu(mem=4G, cores=8)"


@define(kw_only=True)
class Learner:
    validation_size: int = 500

    steps_per_epoch: int = 32
    """Number of steps (batches) per epoch"""

    validation_interval: int = field(default=32)
    validation_top_k: int = 1000

    batch_size: int = 64
    max_epochs: int = 3200
    num_warmup_steps: int = 10000
    warmup_min_factor: float = 0
    lr: float = 3.0e-6
    requirements: str = "duration=4 days & cuda(mem=24G) * 2"
    scheduler: bool = True

    @validation_interval.validator
    def check_validation_interval(self, attribute, value):
        assert self.max_epochs % value == 0, (
            f"Number of epochs ({self.max_epochs}) is not a multiple "
            f"of validation interval ({value})"
        )


@define(kw_only=True)
class Retrieval:
    k: int = 1000
    batch_size: int = 512
    requirements: str = "duration=2 days & cuda(mem=24G)"


@define(kw_only=True)
class Monobert(RerankerMSMarcoV1Configuration):

    indexation: Indexation = Factory(Indexation)
    monobert: Learner = Factory(Learner)
    retrieval: Retrieval = Factory(Retrieval)
