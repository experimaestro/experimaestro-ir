from collections import defaultdict
import itertools
from pathlib import Path
from typing import List, Optional
import random
from datamaestro_text.data.ir.base import GenericTopic, GenericDocument
from experimaestro.notifications import TaskEnv
import torch
from xpmir.learning.context import TrainerContext
from xpmir.rankers import ScoredDocument
from xpmir.rankers.full import FullRetrieverRescorer
from xpmir.neural.dual import DualRepresentationScorer
from xpmir.test.utils.utils import SampleDocumentStore


class ListWrapper(list):
    device = None

    def to(self, device):
        return self

    def __getitem__(self, item):
        return ListWrapper(list.__getitem__(self, item))


class CachedRandomScorer(DualRepresentationScorer):
    def _initialize(self, _random):
        self._cache = defaultdict(lambda: random.uniform(0, 1))

    def cache(self, query: GenericTopic, document: GenericDocument):
        return self._cache[(query.get_text(), document.get_text())]

    def encode(self, texts: List[str]):
        return ListWrapper(texts)

    def score_pairs(
        self, queries, documents, info: Optional[TrainerContext]
    ) -> torch.Tensor:
        scores = [self.cache(q, d) for q, d in zip(queries, documents)]
        return torch.DoubleTensor(scores)

    def score_product(
        self, queries, documents, info: Optional[TrainerContext]
    ) -> torch.Tensor:
        scores = []
        for q in queries:
            scores.append(
                [
                    self.cache(GenericTopic(0, q), GenericDocument(0, d))
                    for d in documents
                ]
            )

        return torch.DoubleTensor(scores)

    def merge_queries(self, queries):
        return ListWrapper(itertools.chain(*queries))


class _FullRetrieverRescorer(FullRetrieverRescorer):
    def retrieve(self, query: str):
        topic = GenericTopic(0, query)
        scored_documents = [
            # Randomly get a score (and cache it)
            ScoredDocument(d, self.scorer.cache(topic, d))
            for d in self.documents
        ]
        scored_documents.sort(reverse=True)
        return scored_documents


def test_fullretrieverescorer(tmp_path: Path):
    NUM_DOCS = 7
    NUM_QUERIES = 9
    TaskEnv.instance().taskpath = tmp_path

    documents = SampleDocumentStore(num_docs=NUM_DOCS)
    scorer = CachedRandomScorer()
    retriever = _FullRetrieverRescorer(documents=documents, scorer=scorer, batchsize=20)

    _retriever = retriever.instance()
    _retriever.initialize()

    # Retrieve normally
    scoredDocuments = {}
    queries = {i: f"Query {i}" for i in range(NUM_QUERIES)}

    # Retrieve query per query
    for qid, query in queries.items():
        scoredDocuments[qid] = _retriever.retrieve(query)

    # Retrieve with batching
    all_results = _retriever.retrieve_all(queries)

    for qid, results in all_results.items():
        expected = scoredDocuments[qid]
        results.sort(reverse=True)
        expected.sort(reverse=True)

        assert [d.document.get_id() for d in expected] == [
            d.document.get_id() for d in results
        ], "Document IDs do not match"
        assert [d.score for d in expected] == [
            d.score for d in results
        ], "Scores do not match"
