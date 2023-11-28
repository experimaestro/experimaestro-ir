# Utility functions for MS-Marco experiments

from typing import Union

from experimaestro import Launcher

from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import (
    ShuffledTrainingTripletsLines,
    StoreTrainingTripletTopicAdapter,
    StoreTrainingTripletDocumentAdapter,
)
from datamaestro_text.data.ir import Documents, Adhoc

from xpmir.utils.functools import cache
from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.letor.samplers import TripletBasedSampler
from xpmir.datasets.adapters import MemoryTopicStore
from xpmir.letor.distillation.samplers import (
    DistillationPairwiseSampler,
    PairwiseHydrator,
)

from xpmir.measures import AP, RR, P, nDCG, Success
from xpmir.utils.functools import partial_cache
from xpmir.papers import configuration


@configuration
class ValidationSample:
    seed: int = 123
    size: int = 500


# Factorizes the different versions of the Callable to avoid redundancy
def prepare_collection(prepare_str: str) -> Union[Documents, Adhoc]:
    return partial_cache(prepare_dataset, prepare_str)()


MEASURES = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10, Success @ 5]

# --- MsMarco v1


@cache
def msmarco_v1_docpairs_sampler(
    *, sample_rate: float = 1.0, sample_max: int = 0, launcher: "Launcher" = None
) -> TripletBasedSampler:
    """Train sampler

    This uses shuffled pre-computed triplets from MS Marco

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
    ).submit(launcher=launcher)

    # Adds the text to the documents
    triplets = StoreTrainingTripletDocumentAdapter(
        data=triplets, store=prepare_collection("irds.msmarco-passage.documents")
    )
    return TripletBasedSampler(source=triplets)


@cache
def msmarco_v1_validation_dataset(cfg: ValidationSample, launcher=None):
    """Sample dev topics to get a validation subset"""
    return RandomFold(
        dataset=prepare_collection("irds.msmarco-passage.dev"),
        seed=cfg.seed,
        fold=0,
        sizes=[cfg.size],
        exclude=prepare_collection("irds.msmarco-passage.dev.small").topics,
    ).submit(launcher=launcher)


@cache
def msmarco_v1_tests(dev_test_size: int = 0):
    """MS-Marco default test collections: DL TREC 2019 & 2020 + devsmall

    devsmall can be restricted to a smaller dataset for debugging using dev_test_size
    """
    v1_devsmall_ds = prepare_collection("irds.msmarco-passage.dev.small")
    if dev_test_size > 0:
        (v1_devsmall_ds,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=v1_devsmall_ds
        )

    return EvaluationsCollection(
        msmarco_dev=Evaluations(v1_devsmall_ds, MEASURES),
        trec2019=Evaluations(
            prepare_dataset("irds.msmarco-passage.trec-dl-2019"), MEASURES
        ),
        trec2020=Evaluations(
            prepare_dataset("irds.msmarco-passage.trec-dl-2020"), MEASURES
        ),
    )


@cache
def msmarco_hofstaetter_ensemble_hard_negatives() -> DistillationPairwiseSampler:
    """Hard negatives from Hofstätter et al. (2020)

    Hard negatives trained by distillation with cross-encoder Improving
    Efficient Neural Ranking Models with Cross-Architecture Knowledge
    Distillation, (Sebastian Hofstätter, Sophia Althammer, Michael Schröder,
    Mete Sertkan, Allan Hanbury), 2020
    """
    train_triples_distil = prepare_dataset(
        "com.github.sebastian-hofstaetter." "neural-ranking-kd.msmarco.ensemble.teacher"
    )

    # Access to topic text
    train_topics = prepare_dataset("irds.msmarco-passage.train.queries")

    # Combine the training triplets with the document and queries texts
    distillation_samples = PairwiseHydrator(
        samples=train_triples_distil,
        documentstore=prepare_collection("irds.msmarco-passage.documents"),
        querystore=MemoryTopicStore(topics=train_topics),
    )

    # Generate a sampler from the samples
    return DistillationPairwiseSampler(samples=distillation_samples)


@cache
def finetuning_validation_dataset(
    cfg: ValidationSample, dataset_id: str, launcher=None
):
    """Sample dev topics to get a validation subset"""
    return RandomFold(
        dataset=prepare_collection(dataset_id),
        seed=cfg.seed,
        fold=0,
        sizes=[cfg.size],
    ).submit(launcher=launcher)
