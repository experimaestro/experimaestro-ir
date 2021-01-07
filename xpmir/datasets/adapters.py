from typing import List
from experimaestro import config, param, cache
from datamaestro_text.data.ir import AdhocAssessments, AdhocTopics


@param("ids", type=List[str])
@param("topics", type=AdhocTopics)
@config()
class AdhocTopicFold(AdhocTopics):
    def iter(self):
        ids = set(self.ids)
        for topic in self.topics.iter():
            if topic.qid in ids:
                yield topic


@param("ids", type=List[str])
@param("qrels", type=AdhocAssessments)
@config()
class AdhocAssessmentFold(AdhocAssessments):
    @cache("assessements.qrels")
    def trecpath(self, path):
        ids = set(self.ids)
        if not path.is_file():
            with path.open("wt") as fp:
                for qrels in self.iter():
                    if qrels.qid in ids:
                        for qrel in qrels.assessments:
                            fp.write(f"""{qrels.qid} 0 {qrel.docno} {qrel.rel}\n""")

        return path

    def iter(self):
        ids = set(self.ids)
        for qrels in self.qrels.iter():
            if qrels.qid in ids:
                yield qrels
