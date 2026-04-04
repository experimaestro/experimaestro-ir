import sys
import torch
import numpy as np
from typing import List
from typing_extensions import TypedDict, ReadOnly
from experimaestro import Param, initializer

from xpm_torch.losses.batchwise import BatchwiseLoss
from xpm_torch.trainers import TrainerContext, LossTrainer

from xpmir.text import TokenizedTexts
from xpmir.letor.records import (
    PairwiseItem,
    ProductItems,
)


class BatchwiseInputs(TypedDict):
    records: ReadOnly[ProductItems]
    tokenized_records: ReadOnly[TokenizedTexts]


def batchwise_collate(records: List[PairwiseItem]) -> BatchwiseInputs:
    """Collate PairwiseItems into a ProductItems batch with in-batch negatives.

    Builds a relevance matrix where the diagonal (positive docs) = 1
    and off-diagonal (other queries' negatives) = 0.
    """
    batch_size = len(records)
    relevances = torch.cat(
        (torch.eye(batch_size), torch.zeros(batch_size, batch_size)), 1
    )

    batch = ProductItems()
    positives = []
    negatives = []
    for record in records:
        batch.add_topics(record.query)
        positives.append(record.positive)
        negatives.append(record.negative)
    batch.add_documents(*positives)
    batch.add_documents(*negatives)
    batch.set_relevances(relevances)
    return BatchwiseInputs(records=batch, tokenized_records=None)


class BatchwiseTrainer(LossTrainer):
    """Batchwise trainer

    Arguments:

    lossfn: The loss function to use
    sampler: A batchwise sampler
    """

    lossfn: Param[BatchwiseLoss]
    """A batchwise loss function"""

    @initializer
    def initialize(
        self,
        random: np.random.RandomState,
        context: TrainerContext,
    ):
        super().initialize(random, context)
        self.lossfn.initialize(context)

        dataset = self.sampler.as_dataset()

        if hasattr(self.ranker, "get_tokenizer_fn"):
            tokenization_fn = self.ranker.get_tokenizer_fn()

            def collate_fn_with_tokenization(
                samples: List[PairwiseItem],
            ) -> BatchwiseInputs:
                inputs = batchwise_collate(samples)
                inputs["tokenized_records"] = tokenization_fn(inputs["records"])
                return inputs

            collate_fn = collate_fn_with_tokenization
        else:
            collate_fn = batchwise_collate

        self._create_dataloader(dataset, collate_fn=collate_fn)

    def train_batch(self, inputs: BatchwiseInputs):
        batch = inputs["records"]
        tokenized_records = inputs.get("tokenized_records")

        # Get the next batch and compute the scores for each query/document
        # Get the scores
        if tokenized_records is not None:
            rel_scores = self.ranker(batch, tokenized=tokenized_records)
        else:
            rel_scores = self.ranker(batch)

        if torch.isnan(rel_scores).any() or torch.isinf(rel_scores).any():
            self.logger.error("nan or inf relevance score detected. Aborting.")
            sys.exit(1)

        # Reshape to get the pairs and compute the loss
        batch_scores = rel_scores.reshape(*batch.relevances.shape)
        self.lossfn.process(
            batch_scores, batch.relevances.to(batch_scores.device), self.context
        )
