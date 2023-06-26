# Utility functions for MS-Marco experiments

from typing import Callable
from attrs import Factory
from functools import cache
import logging

from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import (
    ShuffledTrainingTripletsLines,
    StoreTrainingTripletTopicAdapter,
    StoreTrainingTripletDocumentAdapter,
)
from datamaestro_text.data.ir import Documents, Adhoc
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


@configuration()
class DualMSMarcoV1Configuration(NeuralIRExperiment):
    """Configuration for sparse retriever"""

    validation: ValidationSample = Factory(ValidationSample)


# MsMarco v1

v1_passages: Callable[[], Documents] = partial_cache(
    prepare_dataset, "irds.msmarco-passage.documents"
)
v1_devsmall: Callable[[], Adhoc] = partial_cache(
    prepare_dataset, "irds.msmarco-passage.dev.small"
)
v1_dev: Callable[[], Adhoc] = partial_cache(prepare_dataset, "irds.msmarco-passage.dev")
v1_train: Callable[[], Adhoc] = partial_cache(
    prepare_dataset, "irds.msmarco-passage.train"
)
v1_train_judged: Callable[[], Adhoc] = partial_cache(
    prepare_dataset, "irds.msmarco-passage.train.judged"
)
v1_measures = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10]


@cache
def v1_docpairs_sampler(
    *, sample_rate: float = 1.0, sample_max: int = 0
) -> TripletBasedSampler:
    """Train sampler

    By default, this uses shuffled pre-computed triplets from MS Marco

    :param sample_rate: Sample rate for the triplets (default 1)
    """
    topics = prepare_dataset("irds.msmarco-passage.train.queries")
    train_triples = prepare_dataset("irds.msmarco-passage.train.docpairs")
    triplets = ShuffledTrainingTripletsLines(
        seed=123,
        data=StoreTrainingTripletTopicAdapter(data=train_triples, store=topics),
        sample_rate=sample_rate,
        sample_max=sample_max,
        doc_ids=True,
        topic_ids=False,
    ).submit()

    # Adds the text to the documents
    triplets = StoreTrainingTripletDocumentAdapter(data=triplets, store=v1_passages())
    return TripletBasedSampler(source=triplets)


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
def v1_tests(dev_test_size: int = 0):
    """MS-Marco default test collections: DL TREC 2019 & 2020 + devsmall

    devsmall can be restricted to a smaller dataset for debugging using dev_test_size
    """
    v1_devsmall_ds = v1_devsmall()
    if dev_test_size > 0:
        (v1_devsmall_ds,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=v1_devsmall_ds
        )

    return EvaluationsCollection(
        msmarco_dev=Evaluations(v1_devsmall_ds, v1_measures),
        trec2019=Evaluations(
            prepare_dataset("irds.msmarco-passage.trec-dl-2019"), v1_measures
        ),
        trec2020=Evaluations(
            prepare_dataset("irds.msmarco-passage.trec-dl-2020"), v1_measures
        ),
    )
