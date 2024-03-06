import itertools
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import torch
from experimaestro.notifications import TaskEnv
from datamaestro_text.data.ir import TextItem, IDItem, create_record, TopicRecord

from xpmir.learning.context import TrainerContext
from xpmir.neural.dual import DualRepresentationScorer
from xpmir.rankers import ScoredDocument
from xpmir.rankers.full import FullRetrieverRescorer
from xpmir.test.utils.utils import SampleDocumentStore


class ListWrapper(list):
    device = None

    def to(self, device):
        return self

    def __getitem__(self, item):
        return ListWrapper(list.__getitem__(self, item))


class CachedRandomScorer(DualRepresentationScorer[ListWrapper[str], ListWrapper[str]]):
    def __initialize__(self, options):
        super().__initialize__(options)
        self._cache = defaultdict(lambda: random.uniform(0, 1))

    def cache(self, query: str, document: str):
        return self._cache[(query, document)]

    def encode(self, texts: List[str]):
        return ListWrapper(texts)

    def score_pairs(
        self,
        queries: ListWrapper[str],
        documents: ListWrapper[str],
        info: Optional[TrainerContext],
    ) -> torch.Tensor:
        scores = [self.cache(q, d) for q, d in zip(queries, documents)]
        return torch.DoubleTensor(scores)

    def score_product(
        self,
        queries: ListWrapper[str],
        documents: ListWrapper[str],
        info: Optional[TrainerContext],
    ) -> torch.Tensor:
        scores = []
        for q in queries:
            scores.append([self.cache(q, d) for d in documents])

        return torch.DoubleTensor(scores)

    def merge_queries(self, queries):
        return ListWrapper(itertools.chain(*queries))


class _FullRetrieverRescorer(FullRetrieverRescorer):
    def retrieve(self, record: TopicRecord):
        scored_documents = [
            # Randomly get a score (and cache it)
            ScoredDocument(
                d, self.scorer.cache(record[TextItem].text, d[TextItem].text)
            )
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
    queries = {qid: create_record(text=f"Query {qid}") for qid in range(NUM_QUERIES)}

    # Retrieve query per query
    for qid, query in queries.items():
        scoredDocuments[qid] = _retriever.retrieve(query)

    # Retrieve with batching
    all_results = _retriever.retrieve_all(queries)

    for qid, results in all_results.items():
        expected = scoredDocuments[qid]
        results.sort(reverse=True)
        expected.sort(reverse=True)

        assert [d.document[IDItem].id for d in expected] == [
            d.document[IDItem].id for d in results
        ], "Document IDs do not match"
        assert [d.score for d in expected] == [
            d.score for d in results
        ], "Scores do not match"
