# Utility functions for MS-Marco experiments

from typing import Union
from functools import lru_cache

from experimaestro import Launcher

from datamaestro import prepare_dataset
from datamaestro_ir.transforms import (
    ShuffledTrainingTripletsLines,
    StoreTrainingTripletTopicAdapter,
)
from datamaestro_ir.data import Documents, Adhoc

from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.letor.samplers import TripletBasedSampler
from xpmir.datasets.adapters import MemoryTopicStore
from xpmir.letor.distillation.samplers import (
    DistillationNegativesSampler,
    DistillationListwiseSampler,
    DistillationPairwiseSampler,
)
from xpmir.letor.samplers.adapters import SamplerAdapter
from xpmir.letor.processors import StoreHydrator

from xpmir.measures import AP, RR, P, nDCG, Success
from xpmir.papers import configuration


@configuration
class ValidationSample:
    seed: int = 123
    size: int = 500


@lru_cache
def prepare_collection(prepare_str: str) -> Union[Documents, Adhoc]:
    """Prepare a dataset and caches the result"""
    return prepare_dataset(prepare_str)


MEASURES = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10, Success @ 5]

# --- MsMarco v1


@lru_cache
def msmarco_v1_docpairs_efficient_sampler(
    *,
    sample_rate: float = 1.0,
    sample_max: int = 0,
    launcher: "Launcher" = None,
    seed: int = 123,
) -> TripletBasedSampler:
    """Train sampler

    This uses shuffled pre-computed triplets from MS Marco

    :param sample_rate: Sample rate for the triplets (default 1)
    """
    topics = prepare_dataset("com.microsoft.msmarco.passage.train.queries")
    train_triples = prepare_dataset("com.microsoft.msmarco.passage.train.triples.id")
    triplets = ShuffledTrainingTripletsLines.C(
        seed=seed,
        data=StoreTrainingTripletTopicAdapter.C(data=train_triples, store=topics),
        sample_rate=sample_rate,
        sample_max=sample_max,
        doc_ids=True,
        topic_ids=False,
    ).submit(launcher=launcher)

    # Builds the sampler by hydrating documents
    sampler = TripletBasedSampler.C(source=triplets)
    hydrator = StoreHydrator.C(
        documentstore=prepare_collection("com.microsoft.msmarco.passage.documents")
    )

    return SamplerAdapter.C(sampler=sampler, processors=[hydrator])


@lru_cache
def msmarco_v1_validation_dataset(
    cfg: ValidationSample, launcher=None, only_judged=False
):
    """Sample dev topics to get a validation subset
    If only_judged = False, use com.microsoft.msmarco.passage.dev` (by default), which
    contains the unassessed.
    else use com.microsoft.msmarco.passage.dev.judged, which only contains the queries
    that have at least one non 0 qrel
    """
    if only_judged:
        candidate_ds = prepare_collection("com.microsoft.msmarco.passage.dev.judged")
    else:
        candidate_ds = prepare_collection("com.microsoft.msmarco.passage.dev")

    return RandomFold.C(
        dataset=candidate_ds,
        seed=cfg.seed,
        fold=0,
        sizes=[cfg.size],
        exclude=prepare_collection("com.microsoft.msmarco.passage.dev.small").topics,
    ).submit(launcher=launcher)


@lru_cache
def msmarco_v1_tests(dev_test_size: int = 0, only_judged=False):
    """MS-Marco default test collections: DL TREC 2019 & 2020 + devsmall

    devsmall can be restricted to a smaller dataset for debugging using dev_test_size

    If only_judged = False, use full version for TREC DL (by default), which
    contains the unassessed.
    else use the judged only version, which only contains the queries that have
    at least one non 0 qrel
    """
    v1_devsmall_ds = prepare_collection("com.microsoft.msmarco.passage.dev.small")
    if dev_test_size > 0:
        (v1_devsmall_ds,) = RandomFold.folds(
            seed=0, sizes=[dev_test_size], dataset=v1_devsmall_ds
        )
    if only_judged:
        dl19 = prepare_dataset("com.microsoft.msmarco.passage.trec2019.judged")
        dl20 = prepare_dataset("com.microsoft.msmarco.passage.trec2020.judged")
    else:
        dl19 = prepare_dataset("com.microsoft.msmarco.passage.trec2019")
        dl20 = prepare_dataset("com.microsoft.msmarco.passage.trec2020")
    return EvaluationsCollection(
        msmarco_dev=Evaluations(v1_devsmall_ds, MEASURES),
        trec2019=Evaluations(dl19, MEASURES),
        trec2020=Evaluations(dl20, MEASURES),
    )


@lru_cache
def msmarco_hofstaetter_ensemble_hard_negatives() -> SamplerAdapter:
    """Hard negatives from Hofstätter et al. (2020)

    Hard negatives trained by distillation with cross-encoder Improving
    Efficient Neural Ranking Models with Cross-Architecture Knowledge
    Distillation, (Sebastian Hofstätter, Sophia Althammer, Michael Schröder,
    Mete Sertkan, Allan Hanbury), 2020
    """
    train_triples_distil = prepare_dataset(
        "com.github.sebastian-hofstaetter.neural-ranking-kd.msmarco_ensemble_teacher"
    )

    # Access to topic text
    train_topics = prepare_dataset("com.microsoft.msmarco.passage.train.queries")

    # Generate a sampler from the samples, hydrating with stores
    raw_sampler = DistillationPairwiseSampler.C(samples=train_triples_distil)
    hydrator = StoreHydrator.C(
        documentstore=prepare_collection("com.microsoft.msmarco.passage.documents"),
        querystore=MemoryTopicStore.C(topics=train_topics),
    )

    return SamplerAdapter.C(sampler=raw_sampler, processors=[hydrator])


@lru_cache
def msmarco_rankdistillm_colbert_top50() -> SamplerAdapter:
    """Distillation data from RankZephyr reranking ColBERTv2 top 50 on 10k queries of MSMARCO

    Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage
    Re-Ranking, (Ferdinand Schlatt, Maik Fröbe, Harrisen Scells, Shengyao Zhuang, Bevan Koopman,
    Guido Zuccon, Benno Stein, Martin Potthast, Matthias Hagen), 2025
    """
    train_ranks_distil = prepare_dataset(
        "com.github.webis-de."
        "rank-distillm.rankzephyr_colbert_10000_sampled_50_annotated"
    )

    # Access to topic text
    train_topics = prepare_dataset("com.microsoft.msmarco.passage.train.queries")

    # Generate a sampler from the samples, hydrating with stores
    raw_sampler = DistillationListwiseSampler.C(samples=train_ranks_distil)
    hydrator = StoreHydrator.C(
        documentstore=prepare_collection("com.microsoft.msmarco.passage.documents"),
        querystore=MemoryTopicStore.C(topics=train_topics),
    )

    return SamplerAdapter.C(sampler=raw_sampler, processors=[hydrator])


@lru_cache
def msmarco_colbertv2_annotated(passages_per_query: int) -> SamplerAdapter:
    """Top 500 passages for all queries that have at least one relevance judgement
    in the MS MARCO training query set retrieved by ColBERTv2

    Rank-DistiLLM: Closing the Effectiveness Gap Between Cross-Encoders and LLMs for Passage
    Re-Ranking, (Ferdinand Schlatt, Maik Fröbe, Harrisen Scells, Shengyao Zhuang, Bevan Koopman,
    Guido Zuccon, Benno Stein, Martin Potthast, Matthias Hagen), 2025
    """
    train_ranks_distil = prepare_dataset(
        "com.github.webis-de.rank-distillm.msmarco_colbertv2_annotated"
    )

    # Access to topic text
    train_topics = prepare_dataset("com.microsoft.msmarco.passage.train.queries")

    # Generate a sampler from the samples, hydrating with stores
    raw_sampler = DistillationNegativesSampler.C(
        samples=train_ranks_distil, passages_per_query=passages_per_query
    )
    hydrator = StoreHydrator.C(
        documentstore=prepare_collection("com.microsoft.msmarco.passage.documents"),
        querystore=MemoryTopicStore.C(topics=train_topics),
    )

    return SamplerAdapter.C(sampler=raw_sampler, processors=[hydrator])


@lru_cache
def finetuning_validation_dataset(
    cfg: ValidationSample, dataset_id: str, launcher=None
):
    """Sample dev topics to get a validation subset"""
    return RandomFold.C(
        dataset=prepare_collection(dataset_id),
        seed=cfg.seed,
        fold=0,
        sizes=[cfg.size],
    ).submit(launcher=launcher)
