import asyncio
from pathlib import Path
import os
import sys
from shlex import quote as shquote
import re

from datamaestro_text.data.trec import TipsterCollection, AdhocDocuments
from experimaestro import Task, Argument, Typename, PathArgument, parse_commandline, progress
from experimaestro_ir import NAMESPACE

ANSERINI_NS = Typename("ir.anserini")

@Argument("storePositions", default=False)
@Argument("storeDocvectors", default=False)
@Argument("storeRawDocs", default=False)
@Argument("collection", type=AdhocDocuments)
@Argument("threads", default=8)
@PathArgument("index_path", "index")
@Task(ANSERINI_NS("index"), description="Index a collection")
class IndexCollection:
    CLASSPATH = "io.anserini.index.IndexCollection"
    
    def execute(self):
        from pyserini.setup import configure_classpath
        configure_classpath(os.environ["ANSERINI_CLASSPATH"])
        from jnius_config import get_classpath

        command = ["{}/bin/java".format(os.environ["JAVA_HOME"]), "-cp"]
        command.append(":".join(get_classpath()))
        command.append(IndexCollection.CLASSPATH)
        command.extend(["-index", self.index_path, "-threads", self.threads])
      
        if isinstance(self.collection, TipsterCollection):
            command.extend(["-collection", "TrecCollection",  "-generator", "JsoupGenerator", "-input", self.collection.path])

        if self.storePositions:
            command.append("-storePositions")
        if self.storeDocvectors:
            command.append("-storeDocvectors")
        if self.storeRawDocs:
            command.append("-storeRawDocs")



        # Index and keep track of progress through regular expressions
        RE_FILES = re.compile(rb""".*index\.IndexCollection \(IndexCollection.java:\d+\) - (\d+) files found""")
        RE_FILE = re.compile(rb""".*index\.IndexCollection\$LocalIndexerThread \(IndexCollection.java:\d+\).* docs added.""")
        print(RE_FILES)
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
                    print(data.decode("utf-8"))

            await proc.wait()
            sys.exit(proc.returncode)
                
            
        asyncio.run(run([str(s) for s in command]))
