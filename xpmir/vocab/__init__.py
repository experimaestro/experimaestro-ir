import sys
import re
import torch
from typing import List
import torch.nn as nn
from experimaestro import Config
from xpmir.letor.records import TokenizedTexts
from xpmir.utils import EasyLogger


class Vocab(Config, EasyLogger, nn.Module):
    """
    Represents a vocabulary and corresponding neural encoding technique
    (e.g., embedding). This class can also handle the case of a cross-encoding of the
    query-document couple (e.g. BERT with [SEP]).
    """

    name = None
    __has_clstoken__ = False

    def initialize(self):
        # Easy and hacky way to get the device
        self._dummy_params = nn.Parameter(torch.Tensor())

    @property
    def device(self):
        return self._dummy_params.device

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
        maxlen = min(maxlen or 0, max(lens))

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

        return out_tensor.to(self._dummy_params.device), lens

    def batch_tokenize(
        self, texts: List[str], batch_first=True, maxlen=None
    ) -> TokenizedTexts:
        toks = [self.tokenize(text) for text in texts]
        tokids, lens = self.pad_sequences(
            [[self.tok2id(t) for t in tok] for tok in toks],
            batch_first=batch_first,
            maxlen=maxlen,
        )
        return TokenizedTexts(toks, tokids, lens, None)

    def enc_query_doc(
        self, queries: List[str], documents: List[str], d_maxlen=None, q_maxlen=None
    ):
        """
        Returns encoded versions of the query and document from two
        list (same size) of queries and documents

        May be overwritten in subclass to provide contextualized representation, e.g.
        joinly modeling query and document representations in BERT.
        """

        tokenized_queries = self.batch_tokenize(queries, maxlen=q_maxlen)
        tokenized_documents = self.batch_tokenize(documents, maxlen=d_maxlen)
        return (
            tokenized_queries,
            self(tokenized_queries),
            tokenized_documents,
            self(tokenized_documents),
        )

    @property
    def pad_tokenid(self) -> int:
        raise NotImplementedError()

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

    def forward(self, tok_texts: TokenizedTexts):
        """
        Returns embeddings for the given toks.

        tok_texts: tokenized texts
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
        raise NotImplementedError(f"for {self.__class__}")

    def static(self) -> bool:
        """
        Returns True if the representations are static, i.e., not trained. Otherwise False.
        This allows models to know when caching is appropriate.
        """
        return True

    def maxtokens(self) -> float:
        """Maximum number of tokens that can be processed"""
        return sys.maxsize
