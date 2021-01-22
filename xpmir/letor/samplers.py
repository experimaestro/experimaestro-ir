from typing import Iterator, List, NamedTuple, Optional

import numpy as np
from datamaestro_text.data.ir import Adhoc
from experimaestro import Annotated, Param, config, help, param, tqdm
from experimaestro.annotations import cache
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.utils import EasyLogger


class SamplerRecord:
    """A record from a pointwise sampler"""

    # The query
    query: str

    # The document ID
    docid: str

    # Context of the document
    document: str

    # Score
    score: float

    # The relevance
    relevance: Optional[float]

    def __init__(self, query, docid, document, score, relevance):
        self.query = query
        self.docid = docid
        self.document = document
        self.score = score
        self.relevance = relevance


class Records:
    """Records are the objects passed to the module forwards"""

    # The queries
    queries: List[str]

    # The document IDs
    docids: List[str]

    # Text of the documents
    documents: List[str]

    # The scores of the retriever
    scores: List[float]

    # The relevances
    relevances: List[float]

    # Tokenized
    queries_toks: any
    docs_toks: any

    # Lengths (in tokens)
    queries_len: any
    docs_len: any

    # IDs of tokenized version
    queries_tokids: any
    docs_tokids: any

    def __init__(self):
        self.queries = []
        self.docids = []
        self.scores = []
        self.documents = []
        self.relevances = []

    def add(self, record: SamplerRecord):
        self.queries.append(record.query)
        self.docids.append(record.docid)
        self.relevances.append(record.relevance)
        self.documents.append(record.document)
        self.scores.append(record.score)


@config()
class Sampler(EasyLogger):
    """"Abtract data sampler"""

    def initialize(self, random: np.random.RandomState):
        self.random = random

    def record_iter(self) -> Iterator[SamplerRecord]:
        """Returns an iterator over records (query, document, relevance)"""
        raise NotImplementedError()


@param("dataset", type=Adhoc, help="The topics and assessments")
@param("retriever", type=Retriever, help="The retriever")
@config()
class ModelBasedSampler(Sampler):
    """Sampler based on a retriever"""

    relevant_ratio: Annotated[
        int, help("The sampling ratio of relevant to non relevant")
    ] = 0.5
    dataset: Annotated[Adhoc, help("The topics and assessments")]
    retriever: Annotated[Retriever, help("The retriever")]

    def initialize(self, random):
        super().initialize(random)

        self.retriever.initialize()
        self.index = self.retriever.index
        self.pos_records, self.neg_records = self.readrecords()
        self.logger.info(
            "Loaded %d/%d pos/neg records", len(self.pos_records), len(self.neg_records)
        )

    def getdocuments(self, docids: List[str]):
        self._documents = [
            self.retriever.collection().document_text(docid) for docid in docids
        ]

    @cache("run")
    def readrecords(self, runpath):
        pos_records, neg_records = [], []
        if not runpath.is_file():
            self.logger.info("Reading topics and retrieving documents")
            self.logger.info("Caching in %s", runpath)

            # Read the assessments
            self.logger.info("Reading assessments")
            assessments = {}
            for qrels in self.dataset.assessments.iter():
                doc2rel = {}
                assessments[qrels.qid] = doc2rel
                for docid, rel in qrels.assessments:
                    doc2rel[docid] = rel
            self.logger.info("Read assessments for %d topics", len(assessments))

            with runpath.open("wt") as fp:
                self.logger.info("Retrieving documents for each topic")
                queries = []
                for query in self.dataset.topics.iter():
                    queries.append(query)

                skipped = 0
                for query in tqdm(queries):
                    qassessments = assessments.get(query.qid, None) or {}
                    totalrel = sum(rel for docno, rel in qassessments.items())
                    if totalrel == 0:
                        self.logger.debug(
                            "Skipping topic %s (no relevant documents)", query.qid
                        )
                        skipped += 1
                        continue
                    scoredDocuments = self.retriever.retrieve(
                        query.title
                    )  # type: List[ScoredDocument]
                    for rank, sd in enumerate(scoredDocuments):
                        # Get the assessment (assumes not relevant)
                        rel = qassessments.get(sd.docid, 0)
                        (pos_records if rel > 0 else neg_records).append(
                            SamplerRecord(query.title, sd.docid, None, sd.score, rel)
                        )
                        fp.write(
                            f"{query.title if rank == 0 else ''}\t{sd.docid}\t{sd.score}\t{rel}\n"
                        )
                self.logger.info(
                    "Process %d topics (%d skipped)", len(queries), skipped
                )
        else:
            # Read from file
            self.logger.info("Reading records from file %s", runpath)
            with runpath.open("rt") as fp:
                oldtitle = ""
                for line in fp.readlines():
                    title, docno, score, rel = line.rstrip().split("\t")
                    title = title or oldtitle
                    rel = int(rel)

                    (pos_records if rel > 0 else neg_records).append(
                        SamplerRecord(title, docno, None, float(score), rel)
                    )
                    oldtitle = title

        return pos_records, neg_records

    def prepare(self, record: SamplerRecord):
        if record.document is None:
            record.document = self.index.document_text(record.docid)
        return record

    def record_iter(self) -> Iterator[SamplerRecord]:
        npos = len(self.pos_records)
        nneg = len(self.neg_records)
        for i in range(npos + nneg):
            if self.random.random() < self.relevant_ratio:
                yield self.prepare(self.pos_records[self.random.randint(0, npos)])
            else:
                yield self.prepare(self.neg_records[self.random.randint(0, nneg)])


@config()
class TripletBasedSampler(Sampler):
    """Sampler based on a triplet file"""
