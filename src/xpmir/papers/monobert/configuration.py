from attrs import Factory, field
from xpmir.papers import configuration
from xpmir.papers.helpers import LauncherSpecification
from xpmir.papers.helpers.optim import TransformerOptimization
from xpmir.papers.helpers.msmarco import RerankerMSMarcoV1Configuration


@configuration()
class Indexation(LauncherSpecification):
    requirements: str = "duration=6 days & cpu(cores=8)"


@configuration()
class Learner:
    validation_interval: int = field(default=32)
    validation_top_k: int = 1000

    optimization: TransformerOptimization = Factory(TransformerOptimization)
    requirements: str = "duration=4 days & cuda(mem=24G) * 2"
    sample_rate: float = 1.0
    """Sample rate for triplets"""

    sample_max: int = 0
    """Maximum number of samples considered (before shuffling). 0 for no limit."""


@configuration()
class Retrieval:
    k: int = 1000
    batch_size: int = 512
    requirements: str = "duration=2 days & cuda(mem=24G)"


@configuration()
class Preprocessing:
    requirements: str = "duration=12h & cpu(cores=4)"


@configuration()
class Monobert(RerankerMSMarcoV1Configuration):
    indexation: Indexation = Factory(Indexation)
    retrieval: Retrieval = Factory(Retrieval)

    monobert: Learner = Factory(Learner)
    preprocessing: Preprocessing = Factory(Preprocessing)

    dev_test_size: int = 0
    """Development test size (0 to leave it like this)"""

    base: str = "bert-base-uncased"
    """Identifier for the base model"""
