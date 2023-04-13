from datamaestro import prepare_dataset
from xpmir.letor.samplers import Sampler, PairwiseInBatchNegativesSampler
from xpmir.papers.helpers.msmarco import (
    v1_docpairs_sampler,
    v1_passages,
)

from xpmir.utils.functools import cache
from xpmir.datasets.adapters import MemoryTopicStore
from xpmir.letor.distillation.samplers import (
    DistillationPairwiseSampler,
    PairwiseHydrator,
)


@cache
def splade_sampler(self) -> Sampler:
    """Retrurn different types of trainer based on different configuration"""
    # define the trainer based on different dataset
    if self.cfg.learner.dataset == "":
        train_sampler = v1_docpairs_sampler()
        return PairwiseInBatchNegativesSampler(
            sampler=train_sampler
        )  # generating the batchwise from the pairwise

    elif self.cfg.learner.dataset == "bert_hard_negative":
        # hard negatives trained by distillation with cross-encoder
        # Improving Efficient Neural Ranking Models with Cross-Architecture
        # Knowledge Distillation, (Sebastian Hofstätter, Sophia Althammer,
        # Michael Schröder, Mete Sertkan, Allan Hanbury), 2020
        # In the form of Tuple[Query, Tuple[Document, Document]] without text

        return distillation_sampler()


@cache
def distillation_sampler():
    train_triples_distil = prepare_dataset(
        "com.github.sebastian-hofstaetter."
        + "neural-ranking-kd.msmarco.ensemble.teacher"
    )

    # All the query text
    train_topics = prepare_dataset("irds.msmarco-passage.train.queries")

    # Combine the training triplets with the document and queries texts
    distillation_samples = PairwiseHydrator(
        samples=train_triples_distil,
        documentstore=v1_passages(),
        querystore=MemoryTopicStore(topics=train_topics),
    )

    # Generate a sampler from the samples
    return DistillationPairwiseSampler(samples=distillation_samples)
