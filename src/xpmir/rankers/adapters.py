from typing import List, Tuple
from experimaestro import Param
import datamaestro_text.data.ir.base as ir
from xpmir.rankers import Scorer, ScoredDocument
from xpmir.letor.samplers.hydrators import SampleTransform
from xpmir.learning import ModuleInitOptions


class ScorerTransformAdapter(Scorer):

    scorer: Param[Scorer]
    """The original scorer to be transform"""

    adapters: Param[List[SampleTransform]]
    """The list of sample transforms to apply"""

    def __initialize__(self, options: ModuleInitOptions):
        super().__initialize__(options)
        self.scorer.__initialize__(options)

    def eval(self):
        self.scorer.train(False)

    def to(self, device):
        self.scorer.to(device)

    def transform_records(
        self, qry: str, scored_documents: List[ScoredDocument]
    ) -> Tuple[str, List[ScoredDocument]]:
        topics = [ir.TextTopic(qry)]
        docs = [sd.document for sd in scored_documents]

        for adapter in self.adapters:
            topics = adapter.transform_topics(topics) or topics
            docs = adapter.transform_documents(docs) or docs

        qry_text = topics[0].text
        sd_list = [
            ScoredDocument(doc, sd.score) for (doc, sd) in zip(docs, scored_documents)
        ]

        return (qry_text, sd_list)

    def rsv(self, query: str, scored_documents) -> List[ScoredDocument]:
        qry, sd_list = self.transform_records(query, scored_documents)
        return self.scorer.rsv(qry, sd_list)
