from attrs import Factory, field
from xpmir.papers import configuration
from xpmir.papers.helpers import LauncherSpecification
from xpmir.papers.helpers.msmarco import DenseRetrievalMSMarcoV1Configuration


@configuration()
class Indexation(LauncherSpecification):
    requirements: str = "duration=6 days & cpu(mem=4G, cores=8)"


@configuration()
class Learner:
    validation_interval: int = field(default=32)
    validation_top_k: int = 1000

    # check this two value should be the same
    indexing_interval: int = 128
    sampling_interval: int = 128

    indexspec: str = "Flat"

    # optimization: TransformerOptimization = Factory(TransformerOptimization)
    requirements: str = "duration=4 days & cuda(mem=24G) * 2"
    pass


@configuration()
class Retrieval:
    k: int = 1000
    # batch_size: int = 512
    requirements: str = "duration=2 days & cuda(mem=24G)"
    pass


@configuration()
class ANCE(DenseRetrievalMSMarcoV1Configuration):
    indexation: Indexation = Factory(Indexation)
    retrieval: Retrieval = Factory(Retrieval)

    ance: Learner = Factory(Learner)
