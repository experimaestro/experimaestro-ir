# This package contains all rankers
from abc import ABC, abstractmethod
from experimaestro import tqdm
from typing import (
    Dict,
    List,
    Optional,
    TYPE_CHECKING,
)
from experimaestro import Param, Config, field
from datamaestro_ir.data import (
    Documents,
    DocumentStore,
    IDTextRecord,
    AdhocRun,
)
from xpm_torch import ModuleContainer
from datamaestro_ir.data.base import ScoredDocument

if TYPE_CHECKING:
    pass

import logging

logger = logging.getLogger(__name__)


class Retriever(Config, ModuleContainer, ABC):
    """A retriever is a model to return top-scored documents given a query"""

    store: Param[Optional[DocumentStore]] = field(default=None, ignore_default=True)
    """Give the document store associated with this retriever"""

    def initialize(self):
        pass

    def collection(self):
        """Returns the document collection object"""
        raise NotImplementedError()

    def retrieve_all(
        self, queries: Dict[str, IDTextRecord]
    ) -> Dict[str, List[ScoredDocument]]:
        """Retrieves for a set of documents

        By default, iterate using `self.retrieve`, but this leaves some room open
        for optimization

        Args:

            queries: A dictionary where the key is the ID of the query, and the value
                is the text
        """
        results = {}
        for key, record in tqdm(list(queries.items())):
            results[key] = self.retrieve(record)
        return results

    @abstractmethod
    def retrieve(self, record: IDTextRecord) -> List[ScoredDocument]:
        """Retrieves documents, returning a list sorted by decreasing score

        if `content` is true, includes the document full text
        """
        ...

    def _store(self) -> Optional[DocumentStore]:
        """Returns the associated document store (if any) that can be
        used to get the full text of the documents"""

    def get_store(self) -> Optional[DocumentStore]:
        return self.store or self._store()


class RetrieverHydrator(Retriever):
    """Hydrate retrieved results with document text"""

    retriever: Param[Retriever]
    """The retriever to hydrate"""

    store: Param[DocumentStore]
    """The store for document texts"""

    def initialize(self):
        return self.retriever.initialize()

    def retrieve(self, record: IDTextRecord) -> List[ScoredDocument]:
        return [
            ScoredDocument(self.store.document_ext(sd.document["id"]), sd.score)
            for sd in self.retriever.retrieve(record)
        ]


class RunRetriever(Retriever):
    """A retriever that returns documents from a pre-computed run
    Can be useful to build a two-stage retriever with precomputed first stage (e.g for validation when training a scorer model)
    """

    run: Param[AdhocRun]
    """The pre-computed run"""

    documents: Param[Documents]
    """Associated documents"""

    def initialize(self):
        super().initialize()
        self._run_dict = self.run.get_dict()

    def collection(self):
        return self.documents

    def retrieve(self, record: IDTextRecord) -> List[ScoredDocument]:
        qid = record["id"]
        results = self._run_dict.get(qid, {})

        # Sort by score descending
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

        # Hydrate documents if the documents object is a store
        if isinstance(self.documents, DocumentStore):
            doc_ids = [doc_id for doc_id, _ in sorted_results]
            hydrated_docs = self.documents.documents_ext(doc_ids)
            return [
                ScoredDocument(doc, float(score))
                for doc, (_, score) in zip(hydrated_docs, sorted_results)
            ]

        # Fallback to only ID
        return [
            ScoredDocument({"id": doc_id}, float(score))
            for doc_id, score in sorted_results
        ]

    def _store(self) -> Optional[DocumentStore]:
        return self.documents if isinstance(self.documents, DocumentStore) else None
