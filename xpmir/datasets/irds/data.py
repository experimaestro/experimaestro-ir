from collections import defaultdict
from pathlib import Path
from typing import Iterator, Tuple
from datamaestro_text.interfaces.trec import parse_qrels
from experimaestro import Config, Option
import datamaestro_text.data.ir as ir
from datamaestro_text.interfaces.plaintext import read_tsv
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
        for query in ds.queries_iter():
            yield ir.AdhocTopic(query.query_id, query.text)


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


class TrainingTripletsLines(ir.TrainingTripletsLines, IRDSId):
    irds: Option[str]

    def iter(self) -> Iterator[Tuple[str, str, str]]:
        yield from read_tsv(self.path)
