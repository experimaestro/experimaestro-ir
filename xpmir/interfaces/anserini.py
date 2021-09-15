import asyncio
from pathlib import Path
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
from typing import List
from experimaestro import tqdm as xpmtqdm
import itertools

import datamaestro_text.data.ir.csv as ir_csv
from datamaestro_text.data.ir.trec import (
    AdhocDocuments,
    AdhocTopics,
    TipsterCollection,
    TrecAdhocTopics,
)
from experimaestro import Param, param, pathoption, progress, task
from tqdm import tqdm
from xpmir.index.anserini import Index
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.rankers.standard import BM25, Model
from xpmir.utils import Handler, StreamGenerator


def anserini_classpath():
    import pyserini

    base = Path(pyserini.__file__).parent

    paths = [path for path in base.rglob("anserini-*-fatjar.jar")]
    if not paths:
        raise Exception(f"No matching jar file found in {base}")

    latest = max(paths, key=os.path.getctime)
    return latest


def javacommand():
    """Returns the start of the java command including the Anserini class path"""
    from jnius_config import get_classpath

    command = ["{}/bin/java".format(os.environ["JAVA_HOME"]), "-cp"]
    command.append(":".join(get_classpath() + [str(anserini_classpath())]))

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

        chandler = Handler()

        @chandler()
        def trec_collection(documents: TipsterCollection):
            return contextlib.nullcontext("void"), [
                "-collection",
                "TrecCollection",
                "-input",
                documents.path,
            ]

        @chandler()
        def csv_collection(documents: ir_csv.AdhocDocuments):
            def _generator(out):
                counter = 0
                size = os.path.getsize(documents.path)
                with documents.path.open("rt", encoding="utf-8") as fp, tqdm(
                    total=size, unit="B", unit_scale=True
                ) as pb:
                    for ix, line in enumerate(fp):
                        # Update progress (TODO: cleanup/factorize the progress code)
                        ll = len(line)
                        pb.update(ll)
                        counter += ll
                        progress(counter / size)

                        # Generate document
                        docid, text = line.strip().split(documents.separator, 1)
                        json.dump({"id": docid, "contents": text}, out)
                        out.write("\n")

            generator = StreamGenerator(_generator, mode="wt")

            return generator, [
                "-collection",
                "JsonCollection",
                "-input",
                generator.filepath.parent,
            ]

        @chandler.default()
        def generic_collection(documents: AdhocDocuments):
            """Generic collection handler, supposes that we can iterate documents"""

            def _generator(out):
                for document in xpmtqdm(
                    documents.iter(), unit="documents", total=documents.count
                ):
                    # Generate document
                    json.dump({"id": document.docid, "contents": document.text}, out)
                    out.write("\n")

            generator = StreamGenerator(_generator, mode="wt")

            return generator, [
                "-collection",
                "JsonCollection",
                "-input",
                generator.filepath.parent,
            ]

        generator, args = chandler[self.documents]
        command.extend(args)

        if self.storePositions:
            command.append("-storePositions")
        if self.storeDocvectors:
            command.append("-storeDocvectors")
        if self.storeRaw:
            command.append("-storeRawDocs")
        if self.storeContents:
            command.append("-storeContents")

        print("Running", command)
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
            with generator as _:
                logging.info("Running with command %s", command)
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
@task()
class SearchCollection:
    def execute(self):
        command = javacommand()
        command.append("io.anserini.search.SearchCollection")
        command.extend(("-index", self.index.path, "-output", self.path))

        # Topics

        topicshandler = Handler()

        @topicshandler()
        def trectopics(topics: TrecAdhocTopics):
            return ("-topicreader", "Trec", "-topics", topics.path)

        @topicshandler()
        def tsvtopics(topics: ir_csv.AdhocTopics):
            return ("-topicreader", "TsvInt", "-topics", topics.path)

        command.extend(topicshandler[self.topics])

        # Model

        modelhandler = Handler()

        @modelhandler()
        def handle(bm25: BM25):
            return ("-bm25", "-bm25.k1", str(bm25.k1), "-bm25.b", str(bm25.b))

        command.extend(modelhandler[self.model])

        # Start
        logging.info("Starting command %s", command)
        p = subprocess.run(command)
        sys.exit(p.returncode)


class AnseriniRetriever(Retriever):
    """An Anserini-based retriever

    Attributes:
        index: The Anserini index
        model: the model used to search. Only suupports BM25 so far.
        k: Number of results to retrieve
    """

    index: Param[Index]
    model: Param[Model]
    k: Param[int] = 1500

    def __postinit__(self):
        from pyserini.search import SimpleSearcher

        self.searcher = SimpleSearcher(str(self.index.path))

        modelhandler = Handler()

        @modelhandler()
        def handle(bm25: BM25):
            self.searcher.set_bm25(bm25.k1, bm25.b)

        modelhandler[self.model]

    def getindex(self) -> Index:
        """Returns the associated index (if any)"""
        return self.index

    def retrieve(self, query: str) -> List[ScoredDocument]:
        hits = self.searcher.search(query, k=self.k)
        return [ScoredDocument(hit.docid, hit.score, hit.contents) for hit in hits]
