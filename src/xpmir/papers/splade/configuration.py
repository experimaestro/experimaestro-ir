from attrs import define, Factory, field
from .pipelines import SPLADEMSMarcoV1Configuration


@define(kw_only=True)
class Indexation:
    requirements: str = "duration=2 days & cpu(mem=2G)"
    training_requirements: str = "duration=4 days & cuda(mem=24G)"


@define(kw_only=True)
class Learner:
    model: str = "splade_max"
    """The model to use for training"""

    dataset: str = ""
    """The composition of training pairs, default value represent the
    doc_pair from ir-dataset"""

    validation_size: int = 500
    """Number of topics in the validation set"""

    steps_per_epoch: int = 128
    """Number of steps (batches) per epoch"""

    validation_interval: int = field(default=8)
    """Validation interval (in epochs)"""

    splade_batch_size: int = 124
    """the batch_size for training the splade model"""

    max_epochs: int = 1200
    """the max epochs to train"""

    num_warmup_steps: int = 6000
    """The numbers of the warmup steps during the training"""

    early_stop: int = 0
    """After how many steps without improvement, the trainer stops
       0 means no early stop
    """

    lr: float = 2.0e-5
    """Learning rate for the model"""

    lambda_q: float = 3.0e-4
    """the flop coefficient for query"""

    lambda_d: float = 1.0e-4
    """the flop coefficient for document"""

    lamdba_warmup_steps: int = 50000
    """The numbers of the warmup steps for the lambda to reach the max value"""

    scheduler: bool = True
    """Whether use a scheduler to control the learning rate"""

    requirements: str = "duration=6 days & cuda(mem=24G)"

    @validation_interval.validator
    def check_validation_interval(self, attribute, value):
        assert self.max_epochs % value == 0, (
            f"Number of epochs ({self.max_epochs}) is not a multiple "
            f"of validation interval ({value})"
        )


@define(kw_only=True)
class Evaluation:
    requirements: str = "duration=6 days & cuda(mem=12G)"


@define(kw_only=True)
class BaseRetriever:
    topK: int = 1000
    """How many documents retrieved from the base retriever(bm25)"""


@define(kw_only=True)
class TasBalanceRetriever:
    retTopK: int = 50
    """Top-K when building the validation set(tas-balanced)"""

    indexspec: str = "OPQ4_16,IVF65536_HNSW32,PQ4"
    """faiss index building"""

    faiss_max_traindocs: int = 800_000
    """number of docs for training the index"""


@define(kw_only=True)
class FullRetriever:
    batch_size_full_retriever: int = 200
    """How many documents to be process once to in the
    FullRetrieverScorer(batch_size)"""


@define(kw_only=True)
class SPLADE(SPLADEMSMarcoV1Configuration):

    indexation: Indexation = Factory(Indexation)
    learner: Learner = Factory(Learner)
    evaluation: Evaluation = Factory(Evaluation)
    base_retriever: BaseRetriever = Factory(BaseRetriever)
    tas_balance_retriever: TasBalanceRetriever = Factory(TasBalanceRetriever)
    full_retriever: FullRetriever = Factory(FullRetriever)
