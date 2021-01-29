import sys
import re
from typing import List, Tuple
import torch
from experimaestro import config, param
from xpmir.letor.samplers import Records
from xpmir.utils import EasyLogger


@config()
class Vocab(EasyLogger):
    """
    Represents a vocabulary and corresponding neural encoding technique
    (e.g., embedding). This class can also handle the case of a cross-encoding of the
    query-document couple (e.g. BERT with [SEP]).
    """

    name = None
    __has_clstoken__ = False

    def __postinit__(self):
        pass

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

    def pad_sequences(self, tokensList: List[List[int]], batch_first=True, maxlen=0):
        padding_value = 0
        lens = [len(s) for s in tokensList]
        maxlen = min(maxlen, max(lens)) if maxlen > 0 else max(lens)

        if batch_first:
            out_tensor = torch.full(
                (len(tokensList), maxlen), padding_value, dtype=torch.long
            )
            for i, tokens in enumerate(tokensList):
                out_tensor[i, : lens[i], ...] = torch.LongTensor(tokens[:maxlen])
        else:
            out_tensor = torch.full(
                (maxlen, len(tokensList)), padding_value, dtype=torch.long
            )
            for i, tokens in enumerate(tokensList):
                out_tensor[: lens[i], i, ...] = tokens[:maxlen]

        return out_tensor, lens

    def batch_tokenize(
        self, texts: List[str], batch_first=True, maxlen=0
    ) -> Tuple[List[List[str]], torch.Tensor, List[int]]:
        toks = [self.tokenize(text) for text in texts]
        tokids, lens = self.pad_sequences(
            [[self.tok2id(t) for t in tok] for tok in toks],
            batch_first=batch_first,
            maxlen=maxlen,
        )
        return toks, tokids, lens

    def enc_query_doc(self, queries_tok, documents_tok):
        """
        Returns encoded versions of the query and document from general **inputs dict
        Requires query_tok, doc_tok, query_len, and doc_len.

        May be overwritten in subclass to provide contextualized representation, e.g.
        joinly modeling query and document representations in BERT.
        """
        return {"query": self(queries_tok), "doc": self(documents_tok)}

    def tok2id(self, tok: str) -> int:
        """
        Converts a token to an integer id
        """
        raise NotImplementedError()

    def id2tok(self, idx: int) -> str:
        """
        Converts an integer id to a token
        """
        raise NotImplementedError()

    def lexicon_size(self) -> int:
        """
        Returns the number of items in the lexicon
        """
        raise NotImplementedError()

    def forward(self, toks, lens=None):
        """
        Returns embeddings for the given toks.

        toks: token IDs (shape: [batch, maxlen])
        lens: lengths of each item (shape: [batch])
        """
        raise NotImplementedError()

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

    def maxtokens(self) -> float:
        """Maximum number of tokens that can be processed"""
        return sys.maxsize
