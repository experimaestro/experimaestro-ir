from datamaestro.definitions import data


@data()
class Index:
    """Represents an indexed document collection"""

    def document_text(self, docid: str) -> str:
        """Returns the text of the document"""
        raise NotImplementedError()

    def term_df(self, term: str):
        """Returns the document frequency"""
        raise NotImplementedError()

    @property
    def documentcount(self):
        """Returns the number of documents in the index"""
        raise NotImplementedError()

    @property
    def termcount(self):
        """Returns the number of terms in the index"""
        raise NotImplementedError()
