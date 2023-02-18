from attrs import define
from experimaestro import experiment
from functools import partial
import logging

from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from datamaestro_text.data.ir import AdhocDocuments, Adhoc
from experimaestro.launcherfinder import find_launcher
from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluations, EvaluationsCollection
import xpmir.interfaces.anserini as anserini
from xpmir.letor import Device, Random
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.letor.devices import CudaDevice
from xpmir.letor.optim import (
    TensorboardService,
)
from xpmir.letor.samplers import TripletBasedSampler
from xpmir.measures import AP, RR, P, nDCG
from xpmir.rankers import documents_retriever, RandomScorer
from xpmir.utils.utils import find_java_home


from . import PaperExperiment

logging.basicConfig(level=logging.INFO)


@define(kw_only=True)
class MSMarcoV1Configuration(PaperExperiment):
    gpu: bool = True
    """Use GPU for computation"""


class MSMarcoV1Experiment:
    devsmall: Adhoc
    """A set of 500 topics used for evaluation"""

    dev: Adhoc
    """A set of topics used for validation"""

    tests: EvaluationsCollection
    """The collections on which the models are evaluated"""

    def __init__(self, xp: experiment, cfg: MSMarcoV1Configuration):
        self.cfg = cfg
        self.device = (
            CudaDevice() if cfg.gpu else Device()
        )  #: the device (CPU/GPU) for the experiment

        self.random = Random(seed=0)

        # Datasets: train, validation and test
        self.documents: AdhocDocuments = prepare_dataset(
            "irds.msmarco-passage.documents"
        )  #: MS-Marco document collection
        self.devsmall: Adhoc = prepare_dataset("irds.msmarco-passage.dev.small")
        self.dev: Adhoc = prepare_dataset("irds.msmarco-passage.dev")

        # Defines how we sample train examples
        # (using the shuffled pre-computed triplets from MS Marco)
        train_triples = prepare_dataset("irds.msmarco-passage.train.docpairs")
        triplesid = ShuffledTrainingTripletsLines(
            seed=123,
            data=train_triples,
        ).submit()

        self.train_sampler = TripletBasedSampler(source=triplesid, index=self.documents)

        measures = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10]

        # Prepares the test collections evaluation
        self.tests = EvaluationsCollection(
            msmarco_dev=Evaluations(self.devsmall, measures),
            trec2019=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2019"), measures
            ),
            trec2020=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2020"), measures
            ),
        )

        # Creates the directory with tensorboard data
        self.tb = xp.add_service(TensorboardService(xp.resultspath / "runs"))


@define(kw_only=True)
class RerankerMSMarcoV1Configuration(MSMarcoV1Configuration):
    validation_size: int = 500
    """Number of validation topics"""


class RerankerMSMarcoV1Experiment(MSMarcoV1Experiment):
    """Base class for reranker-based MS-Marco v1 experiments"""

    cfg: RerankerMSMarcoV1Configuration

    def __init__(self, xp: experiment, cfg: RerankerMSMarcoV1Configuration):
        super().__init__(xp, cfg)

        # Launcher for indexation
        self.launcher_index = find_launcher(cfg.indexation.requirements)

        # Sample the dev set to create a validation set
        self.ds_val = RandomFold(
            dataset=self.dev,
            seed=123,
            fold=0,
            sizes=[cfg.validation_size],
            exclude=self.devsmall.topics,
        ).submit()

        # Sets the working directory and the name of the xp
        # Needed by Pyserini
        xp.setenv("JAVA_HOME", find_java_home())

        # Setup indices and validation/test base retrievers
        self.retrievers = partial(
            anserini.retriever,
            anserini.index_builder(launcher=self.launcher_index),
            model=self.basemodel,
        )

        self.model_based_retrievers = partial(
            documents_retriever,
            batch_size=cfg.retrieval.batch_size,
            batcher=PowerAdaptativeBatcher(),
            device=self.device,
        )

        self.test_retrievers = partial(self.retrievers, k=cfg.retrieval.k)

        # Search and evaluate with a random reranker
        random_scorer = RandomScorer(random=self.random).tag("reranker", "random")
        self.tests.evaluate_retriever(
            partial(
                self.model_based_retrievers,
                retrievers=self.test_retrievers,
                scorer=random_scorer,
                device=None,
            )
        )

        # Search and evaluate with the base model
        self.tests.evaluate_retriever(self.test_retrievers, self.launcher_index)