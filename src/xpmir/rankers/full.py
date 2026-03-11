from typing import Any, Dict, List, Tuple

import torch
from datamaestro_ir.data import Documents
from experimaestro import Meta, Param
from experimaestro import tqdm
from xpm_torch.batchers import Batcher

from xpmir.letor.records import DocumentRecord, TopicRecord
from xpmir.neural import DualRepresentationScorer
from xpmir.rankers import Retriever, ScoredDocument


class FullRetriever(Retriever):
    """Retrieves all the documents of the collection

    This can be used to build a small validation set on a subset of the
    collection - in that case, the scorer can be used through a
    TwoStageRetriever, with this retriever as the base retriever.
    """

    documents: Param[Documents]

    def retrieve(self, record: TopicRecord) -> List[ScoredDocument]:
        return [ScoredDocument(doc, 0.0) for doc in self.documents]


class FullRetrieverRescorer(Retriever):
    """Scores all the documents from a collection

    Encodes all queries at once, then processes documents in batches,
    scoring the full query×document matrix each batch. This is more
    efficient than the TwoStageRetriever approach for small collections.
    """

    documents: Param[Documents]
    """The set of documents to consider"""

    scorer: Param[DualRepresentationScorer]
    """The scorer (a dual representation scorer)"""

    batchsize: Param[int] = 0
    batcher: Meta[Batcher] = Batcher.C()

    def initialize(self):
        self.query_batcher = self.batcher.initialize(self.batchsize)
        self.document_batcher = self.batcher.initialize(self.batchsize)

    def encode_queries(self, queries: List[Tuple[str, str]], encoded: List[Any], pbar):
        encoded.append(self.scorer.encode_queries([text for _, text in queries]))
        pbar.update(len(queries))
        return encoded

    def score(
        self,
        documents: List[DocumentRecord],
        queries: List,
        scored_documents: List[List[ScoredDocument]],
        pbar,
    ):
        encoded = self.scorer.encode_documents(documents)

        new_scores = [[] for _ in documents]
        for ix in range(len(queries)):
            query = queries[ix : (ix + 1)]
            scores = self.scorer.score_product(query.to(encoded.device), encoded, None)
            scores = scores.flatten().detach()
            for doc_ix, (document, score) in enumerate(zip(documents, scores)):
                new_scores[doc_ix].append(ScoredDocument(document, float(score)))
                pbar.update(1)

        scored_documents.extend(new_scores)

    def retrieve(self, record: TopicRecord) -> List[ScoredDocument]:
        return self.retrieve_all({"_": record})["_"]

    def retrieve_all(
        self, queries: Dict[str, TopicRecord]
    ) -> Dict[str, List[ScoredDocument]]:
        self.scorer.eval()
        all_queries = list(queries.items())

        with torch.no_grad():
            with tqdm(total=len(all_queries), desc="Encoding queries") as pbar:
                enc_queries = self.query_batcher.reduce(
                    all_queries, self.encode_queries, [], pbar
                )
            enc_queries = self.scorer.merge_queries(enc_queries)

            scored_documents: List[List[ScoredDocument]] = []
            with tqdm(
                total=len(all_queries) * self.documents.documentcount,
                desc="Scoring documents",
            ) as pbar:
                self.document_batcher.process(
                    self.documents, self.score, enc_queries, scored_documents, pbar
                )

        qids = [qid for qid, _ in all_queries]
        return {qid: [sd[ix] for sd in scored_documents] for ix, qid in enumerate(qids)}
