from typing import List, Tuple, Iterable
from experimaestro import Param
from xpmir.rankers import Scorer, ScoredDocument
from xpmir.letor.samplers.hydrators import SampleTransform
from xpmir.letor.records import TopicRecord
from xpmir.learning import ModuleInitOptions


class ScorerTransformAdapter(Scorer):
    """Transforms topic and/or documents output by a scorer when rescoring documents"""

    scorer: Param[Scorer]
    """The original scorer to be transform"""

    adapter: Param[SampleTransform]
    """The list of sample transforms to apply"""

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.scorer.__initialize__(options)

    def eval(self):
        self.scorer.train(False)

    def to(self, device):
        self.scorer.to(device)

    def transform_records(
        self, topic: TopicRecord, scored_documents: Iterable[ScoredDocument]
    ) -> Tuple[TopicRecord, List[ScoredDocument]]:
        topics = [topic.topic]
        docs = [sd.document for sd in scored_documents]

        topics = self.adapter.transform_topics(topics) or topics
        docs = self.adapter.transform_documents(docs) or docs

        sd_list = [
            ScoredDocument(doc, sd.score) for (doc, sd) in zip(docs, scored_documents)
        ]

        return (topics[0].as_record(), sd_list)

    def compute(
        self, topic: TopicRecord, documents: Iterable[ScoredDocument]
    ) -> List[ScoredDocument]:
        topic, sd_list = self.transform_records(topic, documents)
        return self.scorer.compute(topic, sd_list)
