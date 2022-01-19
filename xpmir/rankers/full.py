from typing import List, Optional, Tuple, Dict
from experimaestro import Param, Meta
import torch
from . import Retriever, AdhocDocuments, ScoredDocument, Scorer, AdhocDocument
from xpmir.neural.dual import DualRepresentationScorer
from xpmir.letor.batchers import Batcher


class FullRetriever(Retriever):
    """Retrieves all the documents of the collection

    This can be used to build a small validation set on a subset of the collection - in that
    case, the scorer can be used through a TwoStageRetriever
    """

    documents: Param[AdhocDocuments]

    def retrieve(self, query: str, content=False) -> List[ScoredDocument]:
        if content:
            return [
                ScoredDocument(doc.docid, 0.0, doc.text)
                for doc in self.documents.iter()
            ]
        return [ScoredDocument(docid, 0.0, None) for docid in self.documents.iter_ids()]

    def getReranker(
        self, scorer: Scorer, batch_size: int, batcher: Batcher = Batcher(), device=None
    ):
        if isinstance(scorer, DualRepresentationScorer):
            return FullRetrieverRescorer(
                documents=self.documents,
                scorer=scorer,
                batchsize=batch_size,
                batcher=batcher,
                device=device,
            )
        return super().getReranker(scorer, batch_size, batcher, device)


class FullRetrieverRescorer(Retriever):
    """Scores all the documents from a collection"""

    documents: Param[AdhocDocuments]
    scorer: Param[DualRepresentationScorer]
    batchsize: Param[int] = 0
    batcher: Meta[Batcher] = Batcher()
    device: Meta[Optional[Device]] = None

    def initialize(self):
        self.query_batcher = self.batcher.initialize(self.batchsize)
        self.document_batcher = self.batcher.initialize(self.batchsize)
        self.scorer.initialize(None)

        # Compute with the scorer
        if self.device is not None:
            self.scorer.to(self.device.value)

    def _retrieve(
        self,
        batch: List[ScoredDocument],
        query: str,
        scoredDocuments: List[ScoredDocument],
    ):
        scoredDocuments.extend(self.scorer.rsv(query, batch))

    def encode_queries(
        self,
        queries: List[Tuple[str, str]],
        encoded: List[Tuple[List[str], torch.Tensor]],
    ):
        encoded.append(
            (
                [key for key, _ in queries],
                self.scorer.encode_queries([text for _, text in queries]).to("cpu"),
            )
        )

    def score(
        self,
        documents: List[AdhocDocument],
        queries: List[Tuple[List[str], torch.Tensor]],
        scored_documents: Dict[str, List[ScoredDocument]],
    ):
        docids = [d.docid for d in documents]
        encoded = self.scorer.encode_documents(d.text for d in documents)

        for qids, enc_queries in queries:
            # Returns a query x document matrix
            scores = self.scorer.score_product(
                enc_queries.to(encoded.device), encoded, None
            ).to("cpu")

            # Adds up to the lists
            for ix, d_scores in enumerate(scores):
                for jx, score in enumerate(d_scores):
                    scored_documents.setdefault(qids[ix], []).append(
                        ScoredDocument(docids[jx], float(score))
                    )

    def retrieve_all(self, queries: Dict[str, str]) -> Dict[str, List[ScoredDocument]]:
        self.scorer.eval()
        all_queries = list(queries.items())

        with torch.no_grad():
            # Encode queries
            enc_queries = []
            self.query_batcher.process(all_queries, self.encode_queries, enc_queries)

            # Encode documents and score them
            scored_documents: Dict[str, List[ScoredDocument]] = {}
            self.query_batcher.process(self.documents, enc_queries, scored_documents)

        return scored_documents

    def retrieve(self, query: str):
        return self.retrieve_all({"_": query})["_"]
