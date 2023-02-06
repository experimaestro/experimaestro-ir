from attrs import define, field, Factory
from omegaconf import MISSING
from typing import List


@define(kw_only=True)
class Launcher:
    gpu: bool = True
    tags: List[str] = Factory(list)


@define(kw_only=True)
class Indexation:
    requirements: str = "duration=6 days & cpu(mem=4G, cores=8)"


@define(kw_only=True)
class Learner:
    validation_size: int = 500
    steps_per_epoch: int = 32
    validation_interval: int = 32
    batch_size: int = 64
    max_epoch: int = 3200
    num_warmup_steps: int = 10000
    warmup_min_factor: float = 0
    lr: float = 3.0e-6
    requirements: str = "duration=4 days & cuda(mem=24G) * 2"


@define(kw_only=True)
class Evaluation:
    requirements: str = "duration=2 days & cuda(mem=24G)"


@define(kw_only=True)
class Retrieval:
    k: int = 1000
    val_k: int = 1000


@define(kw_only=True)
class Monobert:
    type: str = MISSING
    launcher: Launcher = field(factory=Launcher)
    indexation: Indexation = MISSING
    learner: Learner = MISSING
    evaluation: Evaluation = MISSING
    retrieval: Retrieval = MISSING
