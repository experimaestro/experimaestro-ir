from typing import Generic, List, Optional, TypeVar
from xpmir.letor.records import BaseRecords, Document, QDMask


Q = TypeVar("Q")
D = TypeVar("D")


class DualRepresentationScorer(Generic[Q, D]):
    """Dual representation scorer

    These scorers compute a representation of documents and queries before
    computing their scores, allowing to optimize the computation
    by avoiding recomputing documents/queries representation
    """

    def computeScores(self, queries: Q, documents: D, mask: Optional[QDMask]):
        """Compute the score for a set of document and query representations, according to the mask"""
        raise NotImplementedError(f"For class {self.__class__}")

    def documentRepresentation(self, documents: List[Document]) -> D:
        raise NotImplementedError(f"For class {self.__class__}")

    def queryRepresentation(self, queries: List[str]) -> Q:
        raise NotImplementedError(f"For class {self.__class__}")

    def _forward(self, inputs: BaseRecords):
        for queries, documents, mask in inputs.structured():
            _documents = self.documentRepresentation([d for d in inputs.documents])
            _queries = self.queryRepresentation([q for q in inputs.queries])

            self.computeScores(_queries, _documents, mask)
