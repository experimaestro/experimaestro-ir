from typing import List, Optional, Tuple, Dict, Any
from experimaestro import Param, Meta, tqdm
import torch
from . import Retriever, ScoredDocument
from datamaestro_text.data.ir import Document, Documents
from xpmir.neural.dual import DualRepresentationScorer
from xpmir.learning.batchers import Batcher
from xpmir.letor import Device


class FullRetriever(Retriever):
    """Retrieves all the documents of the collection

    This can be used to build a small validation set on a subset of the
    collection - in that case, the scorer can be used through a
    TwoStageRetriever, with this retriever as the base retriever.
    """

    documents: Param[Documents]

    def retrieve(self, query: str) -> List[ScoredDocument]:
        return [ScoredDocument(doc, 0.0) for doc in self.documents]


class FullRetrieverRescorer(Retriever):
    """Scores all the documents from a collection"""

    documents: Param[Documents]
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

    def encode_queries(self, queries: List[Tuple[str, str]], encoded: List[Any], pbar):
        """Encode queries and append the tensor of encoded queries to the encoded

        Args:
            queries (List[Tuple[str, str]]): The input queries (id/text)
            encoded (List[Tuple[List[str], torch.Tensor]]): Full list of topics ??
            it should be the List[torch.Tensor]
        """

        encoded.append(self.scorer.encode_queries([text for _, text in queries]))
        pbar.update(len(queries))
        return encoded

    def score(
        self,
        documents: List[Document],
        queries: List,
        scored_documents: List[List[ScoredDocument]],
        pbar,
    ):
        """Score documents for a set of queries

        Every time the score process a batch of document together with whole set
        of queries

        scored_documents is filled with document batches, i.e. it contains [
        [s(q_0, d_0), ..., s(q_n, d0)], ..., [s(q_0, d_m), ..., s(q_n, d_m)] ]
        --> list of m*n

        :param documents: the batch of documents

        :param queries: List of queries

        :param scored_documents: (output) current lists of scored documents (one
            per query)
        """
        # Encode documents
        encoded = self.scorer.encode_documents(d.get_text() for d in documents)

        # Process query by query
        new_scores = [[] for _ in documents]
        for ix in range(len(queries)):
            # Get a range of query records
            query = queries[ix : (ix + 1)]

            # Returns a query x document matrix
            scores = self.scorer.score_product(query.to(encoded.device), encoded, None)

            # Adds up to the lists
            scores = scores.flatten().detach()
            for ix, (document, score) in enumerate(zip(documents, scores)):
                new_scores[ix].append(ScoredDocument(document, float(score)))
                pbar.update(1)

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
            with tqdm(total=len(all_queries), desc="Encoding queries") as pbar:
                enc_queries = self.query_batcher.reduce(
                    all_queries, self.encode_queries, [], pbar
                )
            enc_queries = self.scorer.merge_queries(
                enc_queries
            )  # shape (len(queries), dimension)

            # Encode documents and score them
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
