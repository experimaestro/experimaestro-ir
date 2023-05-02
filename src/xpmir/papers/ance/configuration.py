from attrs import Factory, field
from xpmir.papers import configuration
from xpmir.papers.helpers import LauncherSpecification
from xpmir.papers.helpers.msmarco import DualMSMarcoV1Configuration
from xpmir.papers.helpers.optim import TransformerOptimization


@configuration()
class Indexation(LauncherSpecification):
    requirements: str = "duration=6 days & cpu(mem=4G, cores=8)"
    training_requirements: str = "duration=4 days & cuda(mem=24G)"
    indexspec: str = "Flat"


@configuration()
class Learner:
    validation_interval: int = field(default=32)
    early_stop: int = 0

    # check this two value should be the same
    indexing_interval: int = 128
    sampling_interval: int = 128

    optimization: TransformerOptimization = Factory(TransformerOptimization)
    requirements: str = "duration=4 days & cuda(mem=24G) * 2"


@configuration()
class Retrieval:
    requirements: str = "duration=2 days & cuda(mem=24G)"
    topK: int = 1000
    """How many documents retrieved from the ANCE during the evaluation, also
    for the baseline methods"""

    retTopK: int = 50
    """Top-K when building the validation set"""

    negative_sampler_topk: int = 200
    """How many documents are retrieved when trying to extract the negatives"""

    batch_size_full_retriever: int = 200
    """How many documents to be process once to in the
    FullRetrieverScorer(batch_size)"""


@configuration()
class ANCE(DualMSMarcoV1Configuration):

    indexation: Indexation = Factory(Indexation)
    retrieval: Retrieval = Factory(Retrieval)
    ance: Learner = Factory(Learner)
