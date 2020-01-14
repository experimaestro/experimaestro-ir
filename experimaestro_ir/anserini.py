from pathlib import Path
import os
import sys
import re
import asyncio
import subprocess
import logging

from datamaestro_text.data.trec import TipsterCollection, AdhocDocuments, AdhocTopics, TrecTopics
from experimaestro import task, argument, Identifier, pathargument, parse_commandline, progress
from experimaestro_ir.models import Model, BM25
from experimaestro_ir.utils import Handler
from experimaestro_ir import NAMESPACE
import experimaestro_ir as ir
import experimaestro_ir.trec as trec

ANSERINI_NS = Identifier("ir.anserini")


def javacommand():
    from pyserini.pyclass import configure_classpath
    #os.environ.get("ANSERINI_CLASSPATH", None))
    # configure_classpath()
    from jnius_config import get_classpath

    command = ["{}/bin/java".format(os.environ["JAVA_HOME"]), "-cp"]
    command.append(":".join(get_classpath()))

    return command

@argument("storePositions", default=False)
@argument("storeDocvectors", default=False)
@argument("storeRawDocs", default=False)
@argument("storeTransformedDocs", default=False)
@argument("documents", type=AdhocDocuments)
@argument("threads", default=8, ignored=True)
@pathargument("index_path", "index")
@task(ANSERINI_NS.index, description="Index a documents")
class IndexCollection:
    CLASSPATH = "io.anserini.index.IndexCollection"
    
    def execute(self):
        command = javacommand()
        command.append(IndexCollection.CLASSPATH)
        command.extend(["-index", self.index_path, "-threads", self.threads])
      
        if isinstance(self.documents, TipsterCollection):
            command.extend(["-collection", "TrecCollection",  "-generator", "JsoupGenerator", "-input", self.documents.path])

        if self.storePositions:
            command.append("-storePositions")
        if self.storeDocvectors:
            command.append("-storeDocvectors")
        if self.storeRawDocs:
            command.append("-storeRawDocs")
        if self.storeTransformedDocs:
            command.append("-storeTransformedDocs")



        # Index and keep track of progress through regular expressions
        RE_FILES = re.compile(rb""".*index\.IndexCollection \(IndexCollection.java:\d+\) - (\d+) files found""")
        RE_FILE = re.compile(rb""".*index\.IndexCollection\$LocalIndexerThread \(IndexCollection.java:\d+\).* docs added.""")

        async def run(command):
            proc = await asyncio.create_subprocess_exec(
                *command,
                stderr=None,
                stdout=asyncio.subprocess.PIPE)

            nfiles = -1
            indexedfiles = 0

            while True:
                data = await proc.stdout.readline()
                
                if not data:
                    break

                m = RE_FILES.match(data)
                if m:
                    nfiles = int(m.group(1).decode("utf-8"))
                    print("%d files to index" % nfiles)
                elif RE_FILE.match(data):
                    indexedfiles += 1
                    progress(indexedfiles / nfiles)
                else:
                    sys.stdout.write(data.decode("utf-8"),)

            await proc.wait()
            sys.exit(proc.returncode)
                
            
        asyncio.run(run([str(s) for s in command]))


@argument("index", IndexCollection)
@argument("topics", AdhocTopics)
@argument("model", Model)
@task(ANSERINI_NS.search, trec.TrecSearchResults)
def SearchCollection(index: IndexCollection, topics: AdhocTopics, model: Model, results: Path):
    command = javacommand()
    command.append("io.anserini.search.SearchCollection")
    command.extend(("-index", index.index_path, "-output", results))

    # Topics

    topicshandler = Handler()

    @topicshandler()
    def trectopics(topics: TrecTopics):
        return ("-topicreader", "Trec", "-topics", topics.path)

    command.extend(topicshandler[topics])

    # Model

    modelhandler = Handler()

    @modelhandler()
    def handle(bm25: BM25):
        return ("-bm25", "-k1", str(model.k1), "-b", str(model.b))

    command.extend(modelhandler[model])

    # Start 
    logging.info("Starting command %s", command)
    p = subprocess.run(command)
    sys.exit(p.returncode)