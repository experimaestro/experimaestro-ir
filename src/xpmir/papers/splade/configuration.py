from attrs import Factory, field
from xpmir.papers import configuration
from xpmir.papers.helpers import LauncherSpecification
from xpmir.papers.helpers.msmarco import DualMSMarcoV1Configuration
from xpmir.papers.helpers.optim import TransformerOptimization


@configuration()
class Indexation(LauncherSpecification):
    requirements: str = "duration=2 days & cpu(mem=2G)"

    training_requirements: str = "duration=4 days & cuda(mem=24G)"

    indexspec: str = "OPQ4_16,IVF65536_HNSW32,PQ4"
    """faiss index building"""

    faiss_max_traindocs: int = 800_000
    """number of docs for training the index"""

    max_docs: int = 0
    """Maximum number of indexed documents – should be 0 when not debugging"""


@configuration()
class Learner:
    model: str = "splade_max"
    """The model to use for training"""

    dataset: str = ""
    """The composition of training pairs, default value represent the
    doc_pair from ir-dataset"""

    sample_rate: float = 1.0
    """Sample rate for triplets"""

    sample_max: int = 0
    """Maximum number of samples considered (before shuffling). 0 for no limit."""

    optimization: TransformerOptimization = Factory(TransformerOptimization)

    validation_interval: int = field(default=8)
    """Validation interval (in epochs)"""

    early_stop: int = 0
    """After how many steps without improvement, the trainer stops
       0 means no early stop
    """

    lambda_q: float = 3.0e-4
    """the flop coefficient for query"""

    lambda_d: float = 1.0e-4
    """the flop coefficient for document"""

    lambda_warmup_steps: int = 50000
    """The numbers of the warmup steps for the lambda to reach the max value"""

    requirements: str = "duration=6 days & cuda(mem=24G)"

    # @validation_interval.validator
    # def check_validation_interval(self, attribute, value):
    #     assert self.max_epochs % value == 0, (
    #         f"Number of epochs ({self.max_epochs}) is not a multiple "
    #         f"of validation interval ({value})"
    #     )


@configuration()
class Retrieval:
    requirements: str = "duration=2 days & cuda(mem=24G)"

    topK: int = 1000
    """How many documents retrieved from the SPLADE, also for the baseline
    methods(BM25 and SPLADE in this case)"""

    retTopK: int = 50
    """Top-K when building the validation set(tas-balanced)"""

    batch_size_full_retriever: int = 200
    """How many documents to process at once in the
    FullRetrieverScorer(batch_size)"""

    batch_size_validation_retriever: int = 200
    """How many documents to process at once in the
    FullRetrieverScorer(batch_size) during validation"""


@configuration()
class SPLADE(DualMSMarcoV1Configuration):

    base_hf_id: str = "distilbert-base-uncased"
    indexation: Indexation = Factory(Indexation)
    splade: Learner = Factory(Learner)
    retrieval: Retrieval = Factory(Retrieval)

    dev_test_size: int = 0
    """Development test size (0 to leave it like this)"""
