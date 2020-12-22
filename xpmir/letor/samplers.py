import logging
from typing import Iterator, NamedTuple, Optional
from datamaestro_text.data.ir import Adhoc
from experimaestro import config, param, cache
import numpy as np
from xpmir.letor import Random
from xpmir.rankers import Retriever


class SamplerRecord(NamedTuple):
    query: str
    docid: str
    score: float
    relevance: Optional[float]


@config()
class Sampler:
    """"Abtract data sampler"""

    def initialize(self, random: np.random.RandomState):
        self.random = random

    def record_iter(self) -> Iterator[SamplerRecord]:
        """Returns an iterator over records (query, document, relevance)
        """
        raise NotImplementedError()


@param("dataset", type=Adhoc, help="The topics and qrels")
@param("retriever", type=Retriever, help="The retriever")
@config()
class ModelBasedSampler(Sampler):
    def initialize(self, random):
        super().initialize(random)

        self.retriever.initialize()

        # Read the assessments
        assessments = {}
        for qrels in self.dataset.assessments.iter():
            doc2rel = {}
            assessments[qrels.qid] = doc2rel
            for docid, rel in qrels.assessments:
                doc2rel[docid] = rel

        # Read the topics
        self.records = []
        for query in self.dataset.topics.iter():
            qassessments = assessments.get(query.qid, None) or {}
            for sd in self.retriever.retrieve(query.title):
                rel = qassessments.get(sd.docid, 0)
                self.records.append(SamplerRecord(query.title, sd.docid, sd.score, rel))

    def record_iter(self) -> Iterator[SamplerRecord]:
        # FIXME: should not shuffle in place
        self.random.shuffle(self.records)
        return self.records
