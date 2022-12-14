from typing import List, Optional, Tuple, Dict, Any
from experimaestro import Param, Meta
import torch
from . import Retriever, ScoredDocument
from datamaestro_text.data.ir import AdhocDocument, AdhocDocuments
from xpmir.neural.dual import DualRepresentationScorer
from xpmir.letor.batchers import Batcher
from xpmir.letor import Device


class FullRetriever(Retriever):
    """Retrieves all the documents of the collection

    This can be used to build a small validation set on a subset of the
    collection - in that case, the scorer can be used through a
    TwoStageRetriever
    """

    documents: Param[AdhocDocuments]

    def retrieve(self, query: str, content=False) -> List[ScoredDocument]:
        if content:
            return [
                ScoredDocument(doc.docid, 0.0, doc.text)
                for doc in self.documents.iter()
            ]
        return [ScoredDocument(docid, 0.0, None) for docid in self.documents.iter_ids()]


class FullRetrieverRescorer(Retriever):
    """Scores all the documents from a collection (for a dual representation scorer)"""

    documents: Param[AdhocDocuments]
    """The set of documents to consider"""

    scorer: Param[DualRepresentationScorer]
    """The scorer (a dual representation scorer)"""

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
        encoded: List[Any],
    ):
        """Encode queries and append the tensor of encoded queries to the encoded

        Args:
            queries (List[Tuple[str, str]]): The input queries (id/text)
            encoded (List[Tuple[List[str], torch.Tensor]]): Full list of topics ??
            it should be the List[torch.Tensor]
        """
        encoded.append(self.scorer.encode_queries([text for _, text in queries]))
        return encoded

    def score(
        self,
        documents: List[AdhocDocument],
        queries: List,
        scored_documents: List[List[ScoredDocument]],
    ):
        """Score documents for a set of queries

        Every time the score process a batch of document together with whole set
        of queries

        scored_documents is filled with document batches, i.e. it contains [
        [s(q_0, d_0), ..., s(q_n, d0)], ..., [s(q_0, d_m), ..., s(q_n, d_m)] ]
        --> list of m*n

        Args:
            documents (List[AdhocDocument]): _description_ queries (List): Lis
            of queries scored_documents (List[List[ScoredDocument]]): list of
            scores for each document and for each query (in this order)
        """
        # Encode documents
        docids = [d.docid for d in documents]
        encoded = self.scorer.encode_documents(d.text for d in documents)

        # Process query by query (TODO: improve the process)
        new_scores = [[] for _ in range(len(docids))]
        for ix in range(len(queries)):
            query = queries[ix : (ix + 1)]

            # Returns a query x document matrix
            scores = self.scorer.score_product(query.to(encoded.device), encoded, None)

            # Adds up to the lists
            scores = scores.flatten().detach()
            for docix, score in enumerate(scores):
                new_scores[docix].append(ScoredDocument(docids[docix], float(score)))

        # Add each result to the full document list
        scored_documents.extend(new_scores)

    def retrieve(self, query: str):
        # Only use retrieve_all
        return self.retrieve_all({"_": query})["_"]

    def retrieve_all(self, queries: Dict[str, str]) -> Dict[str, List[ScoredDocument]]:
        """Input is a dictionary of query {id:text},
        return the a dictionary of {query_id: List of ScoredDocuments under the query}
        """

        self.scorer.eval()
        all_queries = list(queries.items())

        with torch.no_grad():
            # Encode all queries
            # each time the batcher will just encode a batchsize of queries
            # and then concat them together
            enc_queries = self.query_batcher.reduce(
                all_queries, self.encode_queries, []
            )
            enc_queries = self.scorer.merge_queries(
                enc_queries
            )  # shape (len(queries), dimension)

            # Encode documents and score them
            scored_documents: List[List[ScoredDocument]] = []
            self.document_batcher.process(
                self.documents, self.score, enc_queries, scored_documents
            )

        qids = [qid for qid, _ in all_queries]
        return {qid: [sd[ix] for sd in scored_documents] for ix, qid in enumerate(qids)}
