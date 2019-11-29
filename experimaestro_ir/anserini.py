from pathlib import Path
import os

from datamaestro_text.data.trec import TipsterCollection, AdhocDocuments
from experimaestro import Task, Argument, Typename, PathArgument, parse_commandline
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
        import pyserini

        command = []
        command.extend(["-index", self.index_path.localpath(), "-threads", self.threads])
      
        if isinstance(self.collection, TipsterCollection):
            command.extend(["-collection", "TrecCollection",  "-generator", "JsoupGenerator", "-input", self.collection.path.localpath()])

        if self.storePositions:
            command.append("-storePositions")
        if self.storeDocvectors:
            command.append("-storeDocvectors")
        if self.storeRawDocs:
            command.append("-storeRawDocs")


        from pyserini.setup import configure_classpath
        configure_classpath(os.environ["ANSERINI_CLASSPATH"])
        from jnius import autoclass

        print(command)        
        indexCollection = autoclass(IndexCollection.CLASSPATH)
        indexCollection.main([str(s) for s in command])

if __name__ == "__main__":
    parse_commandline()
