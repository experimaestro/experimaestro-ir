from pathlib import Path
from cached_property import cached_property
from experimaestro import Choices
from datamaestro.definitions import data, argument
from .base import Index as BaseIndex


@argument("path", type=Path, help="Path to the index")
@argument("storePositions", default=False, help="Store term position within documents")
@argument("storeDocvectors", default=False, help="Store document term vectors")
@argument("storeRaw", default=False, help="Store raw document")
@argument(
    "storeContents",
    default=False,
    help="Store processed documents (e.g. with HTML tags)",
)
@argument(
    "stemmer",
    default="porter",
    checker=Choices(["porter", "krovetz", "none"]),
    help="The stemmer to use",
)
@data()
class Index(BaseIndex):
    _index_reader = None
    _stats = None

    @cached_property
    def index_reader(self):
        from pyserini.index import IndexReader

        return IndexReader(str(self.path))

    def __getstate__(self):
        return {
            key: value for key, value in self.__dict__.items() if key != "index_reader"
        }

    @cached_property
    def documentcount(self):
        return self.index_reader.stats()["documents"]

    @cached_property
    def termcount(self):
        return self.index_reader.stats()["total_terms"]

    def document_text(self, docid):
        doc = self.index_reader.doc(docid)
        return doc.contents()

    def term_df(self, term: str):
        x = self.index_reader.analyze(term)
        if x:
            return self.index_reader.get_term_counts(x[0])[0]
        return 0
