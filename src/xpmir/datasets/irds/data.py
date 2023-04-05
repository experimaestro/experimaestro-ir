import logging
from experimaestro import Config
from experimaestro.compat import cached_property
from typing import Iterator, Tuple
from experimaestro import Option
import datamaestro_text.data.ir as ir
import ir_datasets

# Interface between ir_datasets and datamaestro:
# provides adapted data types


class IRDSId(Config):
    irds: Option[str]
    """The id to load the dataset from ir_datasets"""

    @classmethod
    def __xpmid__(cls):
        return f"ir_datasets.{cls.__qualname__}"

    @cached_property
    def dataset(self):
        return ir_datasets.load(self.irds)


class AdhocTopics(ir.AdhocTopics, IRDSId):
    def iter(self) -> Iterator[ir.AdhocTopic]:
        """Returns an iterator over topics"""
        ds = ir_datasets.load(self.irds)

        from ir_datasets.formats.trec import TrecQuery

        if issubclass(ds.queries_cls(), (TrecQuery,)):

            for query in ds.queries_iter():
                yield ir.AdhocTopic(query.query_id, query.title, {})
        else:
            for query in ds.queries_iter():
                yield ir.AdhocTopic(query.query_id, query.text, {})

    def count(self):
        return self.dataset.queries_count()


class AdhocAssessments(ir.AdhocAssessments, IRDSId):
    def iter(self):
        """Returns an iterator over assessments"""
        ds = ir_datasets.load(self.irds)

        class Qrels(dict):
            def __missing__(self, key):
                qqrel = ir.AdhocAssessedTopic(key, [])
                self[key] = qqrel
                return qqrel

        qrels = Qrels()
        for qrel in ds.qrels_iter():
            qrels[qrel.query_id].assessments.append(
                ir.AdhocAssessment(qrel.doc_id, qrel.relevance)
            )

        return qrels.values()


class AdhocDocuments(ir.AdhocDocumentStore, IRDSId):
    """Wraps an ir datasets collection -- and provide a default text
    value depending on the collection itself"""

    @cached_property
    def _doc_text(self):
        cls = self.dataset.docs_cls()
        if issubclass(cls, ir_datasets.datasets.cord19.Cord19Doc):
            return lambda doc: f"{doc.title}\n{doc.abstract}"
        return lambda doc: doc.text

    def iter(self) -> Iterator[ir.AdhocDocument]:
        """Returns an iterator over adhoc documents"""
        for int_docid, doc in enumerate(self.dataset.docs_iter()):
            yield ir.AdhocDocument(
                doc.doc_id, self._doc_text(doc), internal_docid=int_docid
            )

    @property
    def documentcount(self):
        return self.dataset.docs_count()

    @cached_property
    def store(self):
        return self.dataset.docs_store()

    def document_text(self, docid: str) -> str:
        return self._doc_text(self.store.get(docid))

    def docid_internal2external(self, ix: int):
        return self.dataset.docs_iter()[ix].doc_id

    @cached_property
    def has_title(self):
        return "title" in self.dataset.docs_cls()._fields

    def document(self, ix):
        d = self.dataset.docs_iter()[ix]
        return ir.AdhocDocument(d.doc_id, self._doc_text(d))


class Adhoc(ir.Adhoc, IRDSId):
    pass


class AdhocRun(ir.AdhocRun, IRDSId):
    pass


class TrainingTriplets(ir.TrainingTriplets, IRDSId):
    def iter(self) -> Iterator[Tuple[str, str, str]]:
        ds = ir_datasets.load(self.irds)

        logging.info("Loading queries")
        queries = {}
        for query in ds.queries_iter():
            queries[query.query_id] = query.text

        logging.info("Starting to generate triplets")
        if isinstance(ds.docpairs_handler(), ir_datasets.formats.tsv.TsvDocPairs):
            for entry in ds.docpairs_iter():
                yield (queries[entry.query_id], entry.doc_id_a, entry.doc_id_b)
        logging.info("Ending triplet generation")
