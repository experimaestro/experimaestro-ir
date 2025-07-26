import logging
from datamaestro_text.data.ir.base import (
    IDItem,
    TopicRecord,
    create_record,
)
from experimaestro import Param
from datamaestro_text.data.ir.trec import AdhocRun
from xpmir.rankers import Retriever, ScoredDocument


class AdhocRunRetriever(Retriever):
    """Retrieves using a run file"""

    run: Param[AdhocRun]
    """The run"""

    @property
    def runs(self) -> dict[str, list[ScoredDocument]]:
        result = {}
        for query_id, scored_documents in self.run.get_dict().items():
            scored_documents = list(
                [
                    ScoredDocument(create_record(id=doc_id), score)
                    for doc_id, score in scored_documents.items()
                ]
            )
            scored_documents.sort(key=lambda x: x.score, reverse=True)
            result[query_id] = scored_documents

        return result

    def retrieve(self, record: TopicRecord) -> list[ScoredDocument]:
        scored_documents = self.runs.get(record[IDItem].id, [])
        if len(scored_documents) == 0:
            logging.warning("No documents for topic %s", record[IDItem].id)
        return scored_documents
