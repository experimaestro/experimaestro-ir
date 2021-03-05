from pathlib import Path
from typing import List
from cached_property import cached_property
from experimaestro import Choices, Param, Annotated
from datamaestro.definitions import data
from .base import Index as BaseIndex


@data()
class Index(BaseIndex):
    """Anserini-backed index

    Attributes:

        path: Path to the index
        storePositions: Store term positions
        storeDocvectors: Store document term vectors
        storeRaw: Store raw document
        storeContents: Store processed documents (e.g. with HTML tags)
        stemmer: The stemmer to use
    """

    path: Param[Path]
    storePositions: Param[bool] = False
    storeDocvectors: Param[bool] = False
    storeRaw: Param[bool] = False
    storeContents: Param[bool] = False
    stemmer: Annotated[str, Choices(["porter", "krovetz", "none"])] = "porter"

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

    @cached_property
    def terms(self):
        """Returns a map"""
        return {entry.term: (entry.df, entry.cf) for entry in self.index_reader.terms()}

    def term_df(self, term: str):
        x: List[str] = self.index_reader.analyze(term)
        if x:
            return self.terms.get(x[0], (0, 0))[0]
        return 0
