import re
from torch import nn
from experimaestro import config, param, configmethod
from xpmir.utils import easylog


class VocabEncoder(nn.Module):
    """
    Encodes batches of id sequences
    """

    def __init__(self, vocabulary):
        super().__init__()
        self.vocab = vocabulary

    def forward(self, toks, lens=None):
        """
        Returns embeddings for the given toks.
        toks: token IDs (shape: [batch, maxlen])
        lens: lengths of each item (shape: [batch])
        """
        raise NotImplementedError

    def enc_query_doc(self, queries_tok, documents_tok):
        """
        Returns encoded versions of the query and document from general **inputs dict
        Requires query_tok, doc_tok, query_len, and doc_len.
        May be overwritten in subclass to provide contextualized representation, e.g.
        joinly modeling query and document representations in BERT.
        """
        return {"query": self(queries_tok), "doc": self(documents_tok)}

    def emb_views(self) -> int:
        """
        Returns how many "views" are returned by the embedding layer.
        Most have 1, but sometimes it's useful to return multiple, e.g., BERT's multiple layers
        """
        return 1

    def dim(self) -> int:
        """
        Returns the number of dimensions of the embedding
        """
        raise NotImplementedError()

    def static(self) -> bool:
        """
        Returns True if the representations are static, i.e., not trained. Otherwise False.
        This allows models to know when caching is appropriate.
        """
        return True


@config()
class Vocab:
    """
    Represents a vocabulary and corresponding neural encoding technique (e.g., embedding)
    """

    name = None
    __has_clstoken__ = False

    def __postinit__(self):
        self.logger = easylog()

    def initialize(self):
        pass

    def tokenize(self, text):
        """
        Meant to be overwritten in to provide vocab-specific tokenization when necessary
        e.g., BERT's WordPiece tokenization
        """
        text = text.lower()
        text = re.sub(r"[^a-z0-9]", " ", text)
        return text.split()

    def tok2id(self, tok: str) -> int:
        """
        Converts a token to an integer id
        """
        raise NotImplementedError

    def id2tok(self, idx: int) -> str:
        """
        Converts an integer id to a token
        """
        raise NotImplementedError

    def encoder(self) -> VocabEncoder:
        """
        Encodes batches of id sequences
        """
        raise NotImplementedError

    def lexicon_size(self) -> int:
        """
        Returns the number of items in the lexicon
        """
        raise NotImplementedError
