from typing import Iterator, List
from torch import nn
import numpy as np
from experimaestro import Param
import logging

from xpmir.letor.samplers import PairwiseSampler
from xpmir.letor.records import BaseRecords, PairwiseRecords
from xpmir.neural.generative import DepthUpdatable
from xpmir.letor.trainers import TrainerContext, LossTrainer
from xpmir.learning.context import Loss, StepTrainingHook
from xpmir.utils.utils import foreach, easylog

logger = easylog()


class PairwiseGenerativeLoss(nn.Module, DepthUpdatable):
    """Generic loss for generative models"""

    NAME = "?"

    max_depth: Param[int] = 5
    """The max number of the steps we need to consider, counting from 1"""

    weight: Param[float] = 1.0
    """The weight :math:`w` with which the loss is multiplied (useful when
    combining with other ones)"""

    current_max_depth: int
    """The max_depth for the current learning stage in the progressive training
    stage"""

    def update_depth(self, new_depth):
        if new_depth <= self.max_depth:
            self.current_max_depth = new_depth
            logger.info(
                f"Update the max_depth to {self.current_max_depth} for the loss"
            )
        else:
            self.current_max_depth = self.max_depth

    def initialize(self):
        pass

    def compute(self, records, context):
        pass

    def process(self, records: BaseRecords, context: TrainerContext):
        value = self.compute(records, context)  # tensor shape [bs]
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Loss: {value}")
        context.add_loss(Loss(f"pair-{self.NAME}", value, self.weight))


class GenerativeTrainer(LossTrainer):

    loss: Param[PairwiseGenerativeLoss]

    sampler: Param[PairwiseSampler]
    """The pairwise sampler"""

    def initialize(self, random: np.random.RandomState, context: TrainerContext):
        super().initialize(random, context)
        self.loss.initialize()
        foreach(
            context.hooks(PairwiseGenerativeLoss), lambda loss: loss.initialize()
        )  # maybe later we need to change the sampling target, we can use this hook

        self.sampler.initialize(random)
        self.sampler_iter = self.sampler.pairwise_iter()

    def iter_batches(self) -> Iterator[PairwiseRecords]:
        while True:
            batch = PairwiseRecords()
            for _, record in zip(range(self.batch_size), self.sampler_iter):
                batch.add(record)
            yield batch

    def train_batch(self, records: PairwiseRecords):
        # do the forward pass to get the gradient value
        self.loss.process(records, self.context)


class GenRetDepthUpdateHook(StepTrainingHook):
    """Update the depth of the training instance(loss, scorer, etc) procedure"""

    objects: Param[List[DepthUpdatable]]
    """The objects to update the depth during the learning procedure"""

    update_interval: Param[int] = 200
    """The interval to update the learning depth"""

    def before(self, state: TrainerContext):
        # start with depth 1
        if state.steps % (self.update_interval * state.steps_per_epoch) == 1:
            current_depth = (state.epoch - 1) // self.update_interval + 1
            for object in self.objects:
                object.update_depth(current_depth)


# # to test
# from xpmir.neural.generative.hf import LoadFromT5, T5IdentifierGenerator
# from datamaestro_text.data.ir.base import TextDocument, TextTopic
# from xpmir.letor.records import PairwiseRecord, TopicRecord, DocumentRecord

# if __name__ == "__main__":
#     model = T5IdentifierGenerator(hf_id="t5-base")
#     model.add_pretasks(LoadFromT5(model=model))

#     loss = PairwiseGenerativeRetrievalLoss(id_generator=model, max_depth=5)
#     loss = loss.instance()

#     input = PairwiseRecords()
#     p1 = PairwiseRecord(
#         TopicRecord(TextTopic("query")),
#         DocumentRecord(TextDocument("positive document")),
#         DocumentRecord(TextDocument("negative document")),
#     )
#     p2 = PairwiseRecord(
#         TopicRecord(TextTopic("this is one another")),
#         DocumentRecord(TextDocument("bug please go away")),
#         DocumentRecord(TextDocument("I don't like bugs")),
#     )

#     input.add(p1)
#     input.add(p2)

#     print(loss.compute(input, None))
