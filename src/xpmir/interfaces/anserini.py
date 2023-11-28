import asyncio
from functools import cached_property
from pathlib import Path
import attrs
import contextlib
import json
import logging
import os
import re
import subprocess
import sys
from typing import List, Optional
from experimaestro import tqdm as xpmtqdm, Task, Meta

from datamaestro_text.data.ir import DocumentStore
from datamaestro_text.data.ir.base import IDDocument
import datamaestro_text.data.ir.csv as ir_csv
from datamaestro_text.data.ir.trec import (
    Documents,
    Topics,
    TipsterCollection,
    TrecTopics,
)
from experimaestro import Param, param, pathoption, progress
from tqdm import tqdm
from xpmir.index.anserini import Index
from xpmir.rankers import Retriever, ScoredDocument, document_cache
from xpmir.rankers.standard import BM25, QLDirichlet, Model
from xpmir.utils.utils import Handler, StreamGenerator, needs_java

pyserini_java = needs_java(11)


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


@pyserini_java
@param("documents", type=Documents)
@param("threads", default=8, ignored=True)
@pathoption("path", "index")
class IndexCollection(Index, Task):
    """An [Anserini](https://github.com/castorini/anserini) index"""

    CLASSPATH = "io.anserini.index.IndexCollection"

    documents: Param[Documents]
    """The documents to index"""

    thread: Meta[int] = 8
    """Number of threads when indexing"""

    id: Param[str] = ""
    """Use an empty ID since identifier is determined by documents"""

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
        def csv_collection(documents: ir_csv.Documents):
            def _generator(out):
                counter = 0
                size = os.path.getsize(documents.path)
                with documents.path.open("rt", encoding="utf-8") as fp, tqdm(
                    total=size, unit="B", unit_scale=True
                ) as pb:
                    for ix, line in enumerate(fp):
                        # Update progress
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
        def generic_collection(documents: Documents):
            """Generic collection handler, supposes that we can iterate documents"""

            def _generator(out):
                logging.info(
                    "Starting the iterator over the documents (%d documents)",
                    documents.documentcount,
                )
                for document in xpmtqdm(
                    documents.iter(), unit="documents", total=documents.documentcount
                ):
                    # Generate document
                    json.dump(
                        {"id": document.get_id(), "contents": document.get_text()}, out
                    )
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

        # Index and keep track of progress through regular expressions
        RE_FILES = re.compile(
            rb""".*index\.IndexCollection \(IndexCollection.java:\d+\) - ([\d,]+) files found"""  # noqa: E501
        )
        RE_FILE = re.compile(
            rb""".*index\.IndexCollection\$LocalIndexerThread \(IndexCollection.java:\d+\).* docs added."""  # noqa: E501
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
                        "Did not see the indexing complete log message"
                        " -- exiting with error"
                    )
                    sys.exit(1)
                sys.exit(proc.returncode)

        asyncio.run(run([str(s) for s in command]))


@pyserini_java
@param("index", Index)
@param("topics", Topics)
@param("model", Model)
@pathoption("path", "results.trec")
class SearchCollection(Task):
    def execute(self):
        command = javacommand()
        command.append("io.anserini.search.SearchCollection")
        command.extend(("-index", self.index.path, "-output", self.path))

        # Topics

        topicshandler = Handler()

        @topicshandler()
        def trectopics(topics: TrecTopics):
            return ("-topicreader", "Trec", "-topics", topics.path)

        @topicshandler()
        def tsvtopics(topics: ir_csv.Topics):
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


@attrs.define()
class AnseriniDocument(IDDocument):
    """The hit returned by Anserini"""

    lucene_docid: int
    """Internal document ID"""

    contents: Optional[str] = None
    """Processed content"""

    raw: Optional[str] = None
    """Raw document"""


@pyserini_java
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

    @cached_property
    def searcher(self):
        from pyserini.search.lucene import LuceneSearcher

        searcher = LuceneSearcher(str(self.index.path))

        modelhandler = Handler()

        @modelhandler()
        def handle_bm25(bm25: BM25):
            searcher.set_bm25(bm25.k1, bm25.b)

        @modelhandler()
        def handle_qld(qld: QLDirichlet):
            searcher.set_qld(qld.mu)

        modelhandler[self.model]

        return searcher

    def _get_store(self) -> Optional[Index]:
        """Returns the associated index (if any)"""
        if self.index.storeContents:
            return self.index

    def retrieve(self, query: str) -> List[ScoredDocument]:
        # see
        # https://github.com/castorini/anserini/blob/master/src/main/java/io/anserini/search/SimpleSearcher.java
        hits = self.searcher.search(query, k=self.k)
        store = self.get_store()

        # Batch retrieve documents
        if store is not None:
            return [
                ScoredDocument(doc, hit.score)
                for hit, doc in zip(
                    hits, store.documents_ext([hit.docid for hit in hits])
                )
            ]

        return [
            ScoredDocument(
                AnseriniDocument(hit.docid, hit.lucene_docid, hit.contents, hit.raw)
                if store is None
                else store.document_ext(hit.docid),
                hit.score,
            )
            for hit in hits
        ]


@document_cache
def index_builder(
    documents: Documents, *, launcher=None, **index_params
) -> IndexCollection:
    return IndexCollection(documents=documents, **index_params).submit(
        launcher=launcher
    )


def retriever(
    index_builder: IndexCollection,
    documents: Documents,
    *,
    k: int = None,
    model: Model = None,
    store: DocumentStore = None,
):
    """Function to construct an Anserini retriever"""
    index = index_builder(documents)

    return AnseriniRetriever(
        index=index, k=k or AnseriniRetriever.k, model=model, store=store
    )
