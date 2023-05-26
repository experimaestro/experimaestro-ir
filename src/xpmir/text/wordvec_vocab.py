import pickle
import hashlib
from typing import Optional
import numpy as np
import torch
from pathlib import Path
from torch import nn
from experimaestro import cache, Param

# from onir import vocab, util
from xpmir.letor import Random
from xpmir.letor.records import TokenizedTexts

# from onir.interfaces import wordvec
from xpmir.text import TokensEncoder
from datamaestro_text.data.embeddings import WordEmbeddings

# TODO: add sources to datamaestro _SOURCES = { 'fasttext': {
# 'wiki-news-300d-1M':
#     wordvec.zip_handler('https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'),
#         'crawl-300d-2M':
#         wordvec.zip_handler('https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'),
#     }, 'convknrm': { 'knrm-bing':
#     wordvec.convknrm_handler('http://boston.lti.cs.cmu.edu/appendices/WSDM2018-ConvKNRM/K-NRM/bing/'),
#         'knrm-sogou':
#         wordvec.convknrm_handler('http://boston.lti.cs.cmu.edu/appendices/WSDM2018-ConvKNRM/K-NRM/sogou/'),
#         'convknrm-bing':
#         wordvec.convknrm_handler('http://boston.lti.cs.cmu.edu/appendices/WSDM2018-ConvKNRM/Conv-KNRM/bing/'),
#     'convknrm-sogou':
#     wordvec.convknrm_handler('http://boston.lti.cs.cmu.edu/appendices/WSDM2018-ConvKNRM/Conv-KNRM/sogou/')
#         }, 'bionlp': { 'pubmed-pmc':
#     wordvec.gensim_w2v_handler('http://evexdb.org/pmresources/vec-space-models/PubMed-and-PMC-w2v.bin')
#     }, 'nil': wordvec.nil_handler }


class WordvecVocab(TokensEncoder, nn.Module):
    """Word-based pre-trained embeddings

    Args:

        train: Should the word embeddings be re-retrained?
    """

    data: Param[WordEmbeddings]
    learn: Param[bool] = False
    random: Param[Optional[Random]]

    def __post_init__(self):
        # super().__post_init__()
        self._terms, self._weights = self.load()
        self._term2idx = {t: i for i, t in enumerate(self._terms)}
        matrix = self._weights
        self.size = matrix.shape[1]
        matrix = np.concatenate(
            [np.zeros((1, self.size)), matrix]
        )  # add padding record (-1)
        self.embed = nn.Embedding.from_pretrained(
            torch.from_numpy(matrix.astype(np.float32)), freeze=not self.learn
        )

    @property
    def pad_tokenid(self):
        return 0

    def learnable(self):
        return self.learn

    def __validate__(self):
        """Check that values are coherent"""
        if self.learn:
            assert self.random is not None

    @cache("terms.npy")
    def load(self, path: Path):
        path_lst = path.with_suffix(".lst")
        if path.is_file():
            with path_lst.open("rb") as fp:
                return pickle.load(fp), np.load(path)

        terms, weights = self.data.load()
        np.save(path, weights)
        with path_lst.open("wb") as fp:
            pickle.dump(terms, fp)
        return terms, weights

    def tok2id(self, tok):
        return self._term2idx[tok]

    def id2tok(self, idx):
        return self._terms[idx]

    def lexicon_size(self) -> int:
        return len(self._terms)

    def dim(self):
        return self.embed.weight.shape[0]

    def forward(self, toks: TokenizedTexts):
        # lens ignored
        return self.embed(toks.ids + 1)  # +1 to handle padding at position -1


class WordvecUnkVocab(WordvecVocab):
    """Word-based embeddings with OOV

    A vocabulary in which all unknown terns are given the same token (UNK; 0)
    with random weights
    """

    def __post_init__(self):
        super().__post_init__()
        self._terms = [None] + self._terms
        for term in self._term2idx:
            self._term2idx[term] += 1
        unk_weights = self.random.state.normal(
            scale=0.5, size=(1, self._weights.shape[1])
        )
        self._weights = np.concatenate([unk_weights, self._weights])

    def tok2id(self, tok):
        return self._term2idx.get(tok, 0)

    def lexicon_path_segment(self):
        return "{base}_unk".format(base=super().lexicon_path_segment())

    def lexicon_size(self) -> int:
        return len(self._terms) + 1


class WordvecHashVocab(WordvecVocab):
    """Word-based embeddings with hash-based OOV

    A vocabulary in which all unknown terms are assigned a position in a
    flexible cache based on their hash value. Each position is assigned its own
    random weight.
    """

    hashspace: Param[int] = 1000
    init_stddev: Param[float] = 0.5
    log_miss: Param[bool] = False

    def __initialize__(self, random):
        super().__initialize__()
        hash_weights = random.normal(
            scale=self.init_stddev, size=(self.hashspace, self._weights.shape[1])
        )
        self._weights = np.concatenate([self._weights, hash_weights])

    def tok2id(self, tok):
        try:
            return super().tok2id(tok)
        except KeyError:
            if self.log_miss:
                self.logger.debug(f"vocab miss {tok}")
            # NOTE: use md5 hash (or similar) here because hash() is not
            # consistent across runs
            item = tok.encode()
            item_hash = int(hashlib.md5(item).hexdigest(), 16)
            item_hash_pos = item_hash % self.hashspace
            return len(self._terms) + item_hash_pos

    def lexicon_size(self) -> int:
        return len(self._terms) + self.hashspace
