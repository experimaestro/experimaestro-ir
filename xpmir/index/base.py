from typing import List, Tuple

from experimaestro import Config


class Index(Config):
    """Represents an indexed document collection"""

    def document_text(self, docid: str) -> str:
        """Returns the text of the document given its id"""
        raise NotImplementedError()

    def iter_documents(self) -> List[Tuple[str, str]]:
        """Iterates over (doid, content) couples"""
        raise NotImplementedError()

    def docid_internal2external(self, docid: int):
        raise NotImplementedError()

    def term_df(self, term: str):
        """Returns the document frequency"""
        raise NotImplementedError()

    @property
    def documentcount(self) -> int:
        """Returns the number of documents in the index"""
        raise NotImplementedError()

    @property
    def termcount(self):
        """Returns the number of terms in the index"""
        raise NotImplementedError()
