import sys
import torch
from torch.functional import Tensor
from experimaestro import Param
from typing import List, TypedDict
from typing_extensions import ReadOnly

from xpm_torch.losses.pairwise import PairwiseLoss
from xpm_torch.metrics import ScalarMetric
from xpm_torch.trainers import TrainerContext, LossTrainer

import numpy as np

from xpmir.text import TokenizedTexts
from xpmir.letor.records import (
    PairwiseItem,
    PairwiseItems,
)


class PairwiseInputs(TypedDict):
    records: ReadOnly[PairwiseItems]
    tokenized_records: ReadOnly[TokenizedTexts]


def pairwise_collate(records: List[PairwiseItem]) -> PairwiseInputs:
    """Collate individual PairwiseItems into a PairwiseInputs batch."""
    batch = PairwiseItems()
    for record in records:
        batch.add(record)
    return PairwiseInputs(records=batch, tokenized_records=None)


class PairwiseTrainer(LossTrainer):
    """Pairwise trainer uses samples of the form (query, positive, negative)"""

    lossfn: Param[PairwiseLoss]
    """The loss function"""

    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(self.ranker)
        for hook in context.hooks(PairwiseLoss):
            hook.initialize(self.ranker)
        self.sampler.initialize(random)

        dataset = self.sampler.as_dataset()

        if hasattr(self.ranker, "get_tokenizer_fn"):
            tokenization_fn = self.ranker.get_tokenizer_fn()

            def collate_fn_with_tokenization(
                samples: List[PairwiseItem],
            ) -> PairwiseInputs:
                inputs = pairwise_collate(samples)
                inputs["tokenized_records"] = tokenization_fn(inputs["records"])
                return inputs

            collate_fn = collate_fn_with_tokenization
        else:
            collate_fn = pairwise_collate

        self._create_dataloader(dataset, collate_fn=collate_fn)

    def train_batch(self, inputs: PairwiseInputs):
        records = inputs["records"]
        tokenized_records = inputs.get("tokenized_records")

        # Get the next batch and compute the scores for each query/document
        if tokenized_records is not None:
            rel_scores = self.ranker(
                records, tokenized=tokenized_records, info=self.context
            )
        else:
            rel_scores = self.ranker(records, self.context)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        pairwise_scores = rel_scores.reshape(2, len(records)).T
        self.lossfn.process(pairwise_scores, self.context)

        self.context.add_metric(
            ScalarMetric(
                "accuracy", float(self.acc(pairwise_scores).item()), len(rel_scores)
            )
        )

    def acc(self, scores_by_record) -> Tensor:
        with torch.no_grad():
            return (
                scores_by_record[:, 0] > scores_by_record[:, 1]
            ).sum().float() / len(scores_by_record)
