from experimaestro import Param
from xpmir.neural.generative import ConditionalGenerator
from xpmir.distributed import DistributableModel
from xpmir.letor.records import (
    BaseRecords,
)
from xpmir.learning.context import TrainerContext
from xpmir.rankers import LearnableScorer


class GenerativeCrossScorer(LearnableScorer, DistributableModel):
    """A cross-encoder based on a generative model"""

    #: The pattern used to condition the decoder, with query / document replaced
    # by their string values
    pattern: Param[str] = "Query: {query} Document: {document} Relevant:"

    #: Conditional generator
    generator: Param[ConditionalGenerator]

    #: Relevant token ID
    relevant_token_id: Param[int]

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        # Encode queries and documents
        inputs = [
            self.pattern.format(
                query=tr.topic.get_text(), document=dr.document.get_text()
            )
            for tr, dr in zip(inputs.topics, inputs.documents)
        ]

        step_generator = self.generator.stepwise_iterator()
        step_generator.init(inputs)
        logits = step_generator.step(None)
        return logits.log_softmax(dim=1)[:, self.relevant_token_id]

    def distribute_models(self, update):
        self.generator = update(self.generator)
