from typing import Iterable, List
from pathlib import Path
from experimaestro import Param, Task, cache, pathgenerator, Annotated
from datamaestro_text.data.ir import Adhoc, AdhocAssessments, AdhocTopics
from datamaestro_text.data.ir.trec import TrecAdhocAssessments
from datamaestro_text.data.ir.csv import AdhocTopics as CSVAdhocTopics


class AdhocTopicFold(AdhocTopics):
    ids: Param[List[str]]
    topics: Param[AdhocTopics]

    def iter(self):
        ids = set(self.ids)
        for topic in self.topics.iter():
            if topic.qid in ids:
                yield topic


class AdhocAssessmentFold(AdhocAssessments):
    ids: Param[List[str]]
    qrels: Param[AdhocAssessments]

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


def fold(ids: Iterable[str], dataset: Adhoc):
    """Returns a fold of a dataset, given topic ids"""
    ids = sorted(list(ids))
    topics = AdhocTopicFold(topics=dataset.topics, ids=ids)
    qrels = AdhocAssessmentFold(assessments=dataset.assessments, ids=ids)
    return Adhoc(topics=topics, assessments=qrels, documents=dataset.documents)


class RandomFold(Task):
    """Extracts a random subset of topics from a dataset

    Attributes:
        seed: Random seed used to compute the fold
        size: Number of topics to keep
        dataset: The Adhoc dataset from which a fold is extracted
        inverse: Takes the complement subset
    """

    seed: Param[int]
    size: Param[int]
    dataset: Param[Adhoc]
    complement: Param[bool] = False
    exclude: Param[AdhocTopics]

    assessments: Annotated[Path, pathgenerator("assessments.tsv")]
    topics: Annotated[Path, pathgenerator("topics.tsv")]

    def config(self) -> Adhoc:
        return Adhoc(
            topics=CSVAdhocTopics(path=self.topics),
            assessments=TrecAdhocAssessments(path=self.assessments),
            documents=self.dataset.documents,
        )

    def execute(self):
        import numpy as np

        badids = (
            set(topic.qid for topic in self.exclude.iter()) if self.exclude else set()
        )
        topics = [
            topic for topic in self.dataset.topics.iter() if topic.qid not in badids
        ]
        random = np.random.RandomState(self.seed)
        random.shuffle(topics)
        topics = topics[self.size :] if self.complement else topics[: self.size]

        ids = set()
        self.topics.parent.mkdir(parents=True, exist_ok=True)
        with self.topics.open("wt") as fp:
            for topic in topics:
                ids.add(topic.qid)
                # FIXME: hardcoded title...
                fp.write(f"""{topic.qid}\t{topic.title}\n""")

        with self.assessments.open("wt") as fp:
            for qrels in self.dataset.assessments.iter():
                if qrels.qid in ids:
                    for qrel in qrels.assessments:
                        fp.write(f"""{qrels.qid} 0 {qrel.docno} {qrel.rel}\n""")
