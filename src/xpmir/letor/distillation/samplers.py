from typing import Optional
import numpy as np

from experimaestro import Param, field
from xpmir.rankers import ScoredDocument
from xpm_torch.datasets import (
    LineFileDataset,
    QueryGroupedFileDataset,
    InfiniteDataset,
    ShardedIterableDataset,
    TransformDataset,
)
from xpm_torch.base import Sampler

# Re-export data types from datamaestro_ir
from datamaestro_ir.data.distillation import (  # noqa: F401
    PairwiseDistillationSample,
    PairwiseDistillationSamples,
    PairwiseDistillationSamplesTSV,
    ListwiseDistillationSample,
    ListwiseDistillationSamples,
    ListwiseDistillationSamplesTSV,
    ListwiseDistillationSamplesTSVWithAnnotations,
)


# --- Add as_dataset methods to datamaestro_ir types ---


def _pairwise_tsv_as_dataset(self) -> ShardedIterableDataset:
    """Returns a LineFileDataset that yields PairwiseDistillationSample."""
    return InfiniteDataset(LineFileDataset(self.path, self._parse_line))


PairwiseDistillationSamplesTSV.as_dataset = _pairwise_tsv_as_dataset


def _listwise_tsv_as_dataset(self) -> ShardedIterableDataset:
    """Returns a QueryGroupedFileDataset that yields ListwiseDistillationSample."""
    return InfiniteDataset(
        QueryGroupedFileDataset(
            self.path,
            self._parse_trec_line,
            self._build_group,
            top_k=self.top_k,
        )
    )


ListwiseDistillationSamplesTSV.as_dataset = _listwise_tsv_as_dataset


# --- Sampler classes ---


class DistillationPairwiseSampler(Sampler):
    """Just loops over samples"""

    samples: Param[PairwiseDistillationSamples]

    def initialize(self, random: np.random.RandomState):
        super().initialize(random)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns the underlying dataset for use with StatefulDataLoader."""
        return self.samples.as_dataset()


class DistillationListwiseSampler(Sampler):
    """Just loops over samples"""

    samples: Param[ListwiseDistillationSamples]

    def initialize(self, random: Optional[np.random.RandomState]):
        super().initialize(random)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns the underlying dataset for use with StatefulDataLoader."""
        return self.samples.as_dataset()


class DistillationNegativesSampler(DistillationListwiseSampler):
    """Samples only ``passages_per_query`` documents per query.

    Skips queries that have no relevant document in the retrieved set.

    - Needs relevance judgements to ensure sampling one positive and
      (passages_per_query - 1) negatives per query.
    - Uses ScoredDocument to store relevance labels.
      Note: ignores any scores from the original dataset.
    """

    samples: Param[ListwiseDistillationSamplesTSVWithAnnotations]
    passages_per_query: Param[int] = field(default=8)

    def _sample_docs(self, item):
        qrel = self.samples.qrels_dict.get(item.query["id"], set())
        negatives = []
        positives = []

        for doc in item.documents:
            if doc.document["id"] in qrel:
                positives.append(ScoredDocument(doc.document, score=1))
            else:
                negatives.append(ScoredDocument(doc.document, score=0))

        if not positives:  # this will be skipped by TransformDataset.iter_shard
            return

        # if we have positives, return one per positive doc
        sampled_negatives = [
            negatives[idx]
            for idx in self.random.choice(len(negatives), self.passages_per_query - 1)
        ]

        # return positive document fist and then
        return ListwiseDistillationSample(
            query=item.query, documents=[positives[0]] + sampled_negatives
        )

    def initialize(self, random: Optional[np.random.RandomState]):
        super().initialize(random)

    def as_dataset(self) -> ShardedIterableDataset:
        """Returns the underlying dataset for use with StatefulDataLoader."""
        return TransformDataset(self.samples.as_dataset(), transform=self._sample_docs)
