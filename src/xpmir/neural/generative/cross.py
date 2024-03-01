from experimaestro import Param, Constant
from datamaestro_text.data.ir import TextItem
from xpmir.neural.generative import ConditionalGenerator
from xpmir.letor.records import (
    BaseRecords,
)
from xpmir.learning.context import TrainerContext
from xpmir.rankers import LearnableScorer, ScorerOutputType


class GenerativeCrossScorer(LearnableScorer):
    """A cross-encoder based on a generative model"""

    version: Constant[int] = 2
    """Generative cross scorer version

    changelog:

        1. corrects output type probability
    """

    outputType: ScorerOutputType = ScorerOutputType.LOG_PROBABILITY

    #: The pattern used to condition the decoder, with query / document replaced
    # by their string values
    pattern: Param[str] = "Query: {query} Document: {document} Relevant:"

    #: Conditional generator
    generator: Param[ConditionalGenerator]

    #: Relevant token ID
    relevant_token_id: Param[int]

    def __initialize__(self, options):
        super().__initialize__(options)
        self.generator.initialize(options)

    def forward(self, inputs: BaseRecords, info: TrainerContext = None):
        # Encode queries and documents
        inputs = [
            self.pattern.format(query=tr[TextItem].text, document=dr[TextItem].text)
            for tr, dr in zip(inputs.topics, inputs.documents)
        ]

        step_generator = self.generator.stepwise_iterator()
        step_generator.init(inputs)
        logits = step_generator.step(None)
        return logits.log_softmax(dim=1)[:, self.relevant_token_id]
