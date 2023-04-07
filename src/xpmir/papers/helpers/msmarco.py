# Utility functions for MS-Marco experiments

from typing import Callable
from attrs import Factory
from functools import cache
import logging

from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from datamaestro_text.data.ir import AdhocDocuments, Adhoc
from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.letor.samplers import TripletBasedSampler

from xpmir.measures import AP, RR, P, nDCG
from xpmir.utils.functools import partial_cache
from . import NeuralIRExperiment, configuration

logging.basicConfig(level=logging.INFO)


@configuration
class ValidationSample:
    seed: int = 123
    size: int = 500


@configuration()
class RerankerMSMarcoV1Configuration(NeuralIRExperiment):
    """Configuration for rerankers"""

    validation: ValidationSample = Factory(ValidationSample)


# MsMarco v1

v1_passages: Callable[[], AdhocDocuments] = partial_cache(
    prepare_dataset, "irds.msmarco-passage.documents"
)
v1_devsmall: Callable[[], Adhoc] = partial_cache(
    prepare_dataset, "irds.msmarco-passage.dev.small"
)
v1_dev: Callable[[], Adhoc] = partial_cache(prepare_dataset, "irds.msmarco-passage.dev")
v1_measures = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10]


@cache
def v1_docpairs_sampler() -> TripletBasedSampler:
    """Train sampler

    By default, this uses shuffled pre-computed triplets from MS Marco
    """
    train_triples = prepare_dataset("irds.msmarco-passage.train.docpairs")
    triplets = ShuffledTrainingTripletsLines(
        seed=123,
        data=train_triples,
    ).submit()
    return TripletBasedSampler(source=triplets, index=v1_passages())


@cache
def v1_validation_dataset(cfg: ValidationSample):
    """Sample dev topics to get a validation subset"""
    return RandomFold(
        dataset=v1_dev(),
        seed=cfg.seed,
        fold=0,
        sizes=[cfg.size],
        exclude=v1_devsmall().topics,
    ).submit()


@cache
def v1_tests():
    """MS-Marco default test collections: DL TREC 2019 & 2020 + devsmall"""
    return EvaluationsCollection(
        msmarco_dev=Evaluations(v1_devsmall(), v1_measures),
        trec2019=Evaluations(
            prepare_dataset("irds.msmarco-passage.trec-dl-2019"), v1_measures
        ),
        trec2020=Evaluations(
            prepare_dataset("irds.msmarco-passage.trec-dl-2020"), v1_measures
        ),
    )
