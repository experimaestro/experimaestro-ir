from pathlib import Path
import os
import sys
import re
import asyncio
import subprocess
import logging

import datamaestro_text.data.ir.csv as ir_csv
from datamaestro_text.data.ir.trec import (
    TipsterCollection,
    AdhocDocuments,
    AdhocTopics,
    TrecAdhocTopics,
)
from experimaestro import (
    task,
    param,
    pathoption,
    progress,
    config,
)
from xpmir.letor.samplers import Collection
from xpmir.dm.data.anserini import Index
from xpmir.rankers import Retriever
from xpmir.rankers.standard import Model, BM25
from xpmir.utils import Handler
from xpmir.evaluation import TrecAdhocRun
import xpmir as ir


def javacommand():
    from pyserini.pyclass import configure_classpath

    # os.environ.get("ANSERINI_CLASSPATH", None))
    # configure_classpath()
    from jnius_config import get_classpath

    command = ["{}/bin/java".format(os.environ["JAVA_HOME"]), "-cp"]
    command.append(":".join(get_classpath()))

    return command


@param("documents", type=AdhocDocuments)
@param("threads", default=8, ignored=True)
@pathoption("path", "index")
@task(description="Index a documents")
class IndexCollection(Index):
    """An [Anserini](https://github.com/castorini/anserini) index"""

    CLASSPATH = "io.anserini.index.IndexCollection"

    def execute(self):
        command = javacommand()
        command.append(IndexCollection.CLASSPATH)
        command.extend(["-index", self.path, "-threads", self.threads])

        if isinstance(self.documents, TipsterCollection):
            command.extend(
                [
                    "-collection",
                    "TrecCollection",
                    "-input",
                    self.documents.path,
                ]
            )

        if self.storePositions:
            command.append("-storePositions")
        if self.storeDocvectors:
            command.append("-storeDocvectors")
        if self.storeRaw:
            command.append("-storeRawDocs")
        if self.storeContents:
            command.append("-storeContents")

        print(command)
        # Index and keep track of progress through regular expressions
        RE_FILES = re.compile(
            rb""".*index\.IndexCollection \(IndexCollection.java:\d+\) - ([\d,]+) files found"""
        )
        RE_FILE = re.compile(
            rb""".*index\.IndexCollection\$LocalIndexerThread \(IndexCollection.java:\d+\).* docs added."""
        )
        RE_COMPLETE = re.compile(
            rb""".*IndexCollection\.java.*Indexing Complete.*documents indexed"""
        )

        async def run(command):
            proc = await asyncio.create_subprocess_exec(
                *command, stderr=None, stdout=asyncio.subprocess.PIPE
            )

            nfiles = -1
            indexedfiles = 0
            complete = False

            while True:
                data = await proc.stdout.readline()

                if not data:
                    break

                m = RE_FILES.match(data)
                complete = complete or (RE_COMPLETE.match(data) is not None)
                if m:
                    nfiles = int(m.group(1).decode("utf-8").replace(",", ""))
                    print("%d files to index" % nfiles)
                elif RE_FILE.match(data):
                    indexedfiles += 1
                    progress(indexedfiles / nfiles)
                else:
                    sys.stdout.write(
                        data.decode("utf-8"),
                    )

            await proc.wait()
            if proc.returncode == 0 and not complete:
                logging.error(
                    "Did not see the indexing complete log message -- exiting with error"
                )
                sys.exit(1)
            sys.exit(proc.returncode)

        asyncio.run(run([str(s) for s in command]))


@param("index", Index)
@param("topics", AdhocTopics)
@param("model", Model)
@pathoption("path", "results.trec")
@task(parents=TrecAdhocRun)
def SearchCollection(index: Index, topics: AdhocTopics, model: Model, path: Path):
    command = javacommand()
    command.append("io.anserini.search.SearchCollection")
    command.extend(("-index", index.path, "-output", path))

    # Topics

    topicshandler = Handler()

    @topicshandler()
    def trectopics(topics: TrecAdhocTopics):
        return ("-topicreader", "Trec", "-topics", topics.path)

    @topicshandler()
    def tsvtopics(topics: ir_csv.AdhocTopics):
        return ("-topicreader", "TsvInt", "-topics", topics.path)

    command.extend(topicshandler[topics])

    # Model

    modelhandler = Handler()

    @modelhandler()
    def handle(bm25: BM25):
        return ("-bm25", "-bm25.k1", str(model.k1), "-bm25.b", str(model.b))

    command.extend(modelhandler[model])

    # Start
    logging.info("Starting command %s", command)
    p = subprocess.run(command)
    sys.exit(p.returncode)


@param("index", Index)
@param("model", Model)
@param("k", default=1500, help="Number of results to retrieve")
@config()
class AnseriniRetriever(Retriever):
    def initialize(self):
        from pyserini.search import SimpleSearcher

        self.searcher = SimpleSearcher(str(self.index.path))
        self.searcher.set_bm25(0.9, 0.4)

    def retrieve(self, query: str):
        return [hit for hit in self.searcher.search(query)]


@param("index", type=Index, help="The anserini index")
@config()
class AnseriniCollection(Collection):
    def __postinit__(self):
        from pyserini.index import IndexReader

        self.index_reader = IndexReader(str(self.index.path))
        self.stats = self.index_reader.stats()

    def __getstate__(self):
        return {"index": self.index}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.__postinit__()

    def documentcount(self):
        return self.stats["documents"]

    def termcount(self):
        return self.stats["total_terms"]

    def document_text(self, docid):
        doc = self.index_reader.doc(docid)
        return doc.contents()

    def term_df(self, term: str):
        x = self.index_reader.analyze(term)
        if x:
            return self.index_reader.get_term_counts(x[0])[0]
        return 0
