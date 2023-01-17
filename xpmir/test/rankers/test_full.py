from collections import defaultdict
import itertools
from typing import List, Optional
import random
import torch
from xpmir.letor.context import TrainerContext
from xpmir.rankers import ScoredDocument
from xpmir.rankers.full import FullRetrieverRescorer
from xpmir.neural.dual import DualRepresentationScorer
from xpmir.test.utils.utils import SampleAdhocDocumentStore


class ListWrapper(list):
    device = None

    def to(self, device):
        return self

    def __getitem__(self, item):
        return ListWrapper(list.__getitem__(self, item))


class CachedRandomScorer(DualRepresentationScorer):
    def _initialize(self, _random):
        self.cache = defaultdict(lambda: random.uniform(0, 1))

    def encode(self, texts: List[str]):
        return ListWrapper(texts)

    def score_pairs(
        self, queries, documents, info: Optional[TrainerContext]
    ) -> torch.Tensor:
        scores = [self.cache[(q, d)] for q, d in zip(queries, documents)]
        return torch.DoubleTensor(scores)

    def score_product(
        self, queries, documents, info: Optional[TrainerContext]
    ) -> torch.Tensor:
        scores = []
        for q in queries:
            scores.append([self.cache[(q, d)] for d in documents])

        return torch.DoubleTensor(scores)

    def merge_queries(self, queries):
        return ListWrapper(itertools.chain(*queries))


class _FullRetrieverRescorer(FullRetrieverRescorer):
    def retrieve(self, query: str):
        scored_documents = [
            ScoredDocument(d.docid, self.scorer.cache[(query, d.text)])
            for d in self.documents
        ]
        scored_documents.sort(reverse=True)
        return scored_documents


def test_fullretrieverescorer():
    NUM_DOCS = 7
    NUM_QUERIES = 9

    documents = SampleAdhocDocumentStore(num_docs=NUM_DOCS)
    scorer = CachedRandomScorer()
    retriever = _FullRetrieverRescorer(documents=documents, scorer=scorer, batchsize=20)

    _retriever = retriever.instance()
    _retriever.initialize()

    # Retrieve normally
    scoredDocuments = {}
    queries = {i: f"Query {i}" for i in range(NUM_QUERIES)}

    for qid, query in queries.items():
        scoredDocuments[qid] = _retriever.retrieve(query)

    # Retrieve with batching
    all_results = _retriever.retrieve_all(queries)

    for qid, results in all_results.items():
        expected = scoredDocuments[qid]
        results.sort(reverse=True)
        expected.sort(reverse=True)

        assert [d.docid for d in expected] == [
            d.docid for d in results
        ], "Document IDs do not match"
        assert [d.score for d in expected] == [
            d.score for d in results
        ], "Scores do not match"
