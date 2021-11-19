import logging
from typing import Iterator, Tuple
from experimaestro import Option
import datamaestro_text.data.ir as ir
import ir_datasets

# Interface between ir_datasets and datamaestro:
# provides adapted data types


class IRDSId:
    @classmethod
    def __xpmid__(cls):
        return f"ir_datasets.{cls.__qualname__}"


class AdhocTopics(ir.AdhocTopics, IRDSId):
    irds: Option[str]

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


class AdhocAssessments(ir.AdhocAssessments, IRDSId):
    irds: Option[str]

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


class AdhocDocuments(ir.AdhocDocuments, IRDSId):
    irds: Option[str]

    def iter(self) -> Iterator[ir.AdhocDocument]:
        """Returns an iterator over adhoc documents"""
        ds = ir_datasets.load(self.irds)
        for doc in ds.docs_iter():
            yield ir.AdhocDocument(doc.doc_id, doc.text)


class Adhoc(ir.Adhoc, IRDSId):
    irds: Option[str]


class AdhocRun(ir.AdhocRun, IRDSId):
    irds: Option[str]


class TrainingTriplets(ir.TrainingTriplets, IRDSId):
    irds: Option[str]

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
