from typing import Iterable, List, Optional
from pathlib import Path
from experimaestro import Param, Task, cache, pathgenerator, Annotated
from datamaestro_text.data.ir import Adhoc, AdhocAssessments, AdhocTopics
from datamaestro_text.data.ir.trec import TrecAdhocAssessments
from datamaestro_text.data.ir.csv import AdhocTopics as CSVAdhocTopics
import math


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
        sizes: Number of topics of each fold (or percentage if sums to 1)
        dataset: The Adhoc dataset from which a fold is extracted
        fold: Which fold to take
    """

    seed: Param[int]
    sizes: Param[List[float]]
    dataset: Param[Adhoc]
    fold: Param[int]
    exclude: Param[Optional[AdhocTopics]]

    assessments: Annotated[Path, pathgenerator("assessments.tsv")]
    topics: Annotated[Path, pathgenerator("topics.tsv")]

    def __validate__(self):
        assert self.fold < len(self.sizes)

    @staticmethod
    def folds(
        seed: int,
        sizes: List[float],
        dataset: Param[Adhoc],
        exclude: Param[AdhocTopics] = None,
        submit=True,
    ):
        """Creates folds

        Parameters:

        - submit: if true (default), submits the fold tasks to experimaestro
        """

        folds = []
        for ix in range(len(sizes)):
            fold = RandomFold(
                seed=seed, sizes=sizes, dataset=dataset, exclude=exclude, fold=ix
            )
            if submit:
                fold = fold.submit()
            folds.append(fold)

        return folds

    def config(self) -> Adhoc:
        return Adhoc(
            id="",  # No need to have a more specific id since it is generated
            topics=CSVAdhocTopics(id="", path=self.topics),
            assessments=TrecAdhocAssessments(id="", path=self.assessments),
            documents=self.dataset.documents,
        )

    def execute(self):
        import numpy as np

        # Get topics
        badids = (
            set(topic.qid for topic in self.exclude.iter()) if self.exclude else set()
        )
        topics = [
            topic for topic in self.dataset.topics.iter() if topic.qid not in badids
        ]
        random = np.random.RandomState(self.seed)
        random.shuffle(topics)

        # Get the fold
        sizes = np.array([0.0] + self.sizes)
        s = sizes.sum()
        if abs(s - 1) < 1e-6:
            sizes = np.round(len(topics) * sizes)
            sizes = np.round(len(topics) * sizes / sizes.sum())

        assert sizes[self.fold + 1] > 0

        indices = sizes.cumsum().astype(int)
        topics = topics[indices[self.fold] : indices[self.fold + 1]]

        # Write topics and assessments
        ids = set()
        self.topics.parent.mkdir(parents=True, exist_ok=True)
        with self.topics.open("wt") as fp:
            for topic in topics:
                ids.add(topic.qid)
                fp.write(f"""{topic.qid}\t{topic.text}\n""")

        with self.assessments.open("wt") as fp:
            for qrels in self.dataset.assessments.iter():
                if qrels.qid in ids:
                    for qrel in qrels.assessments:
                        fp.write(f"""{qrels.qid} 0 {qrel.docno} {qrel.rel}\n""")
