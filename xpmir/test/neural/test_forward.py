import logging
from typing import List, Tuple
import pytest
import torch
from xpmir.index.base import Index
from xpmir.letor import Random
from xpmir.letor.records import (
    Document,
    PairwiseRecord,
    PairwiseRecords,
    PointwiseRecord,
    PointwiseRecords,
    ProductRecords,
    Query,
    TokenizedTexts,
)
from xpmir.vocab import Vocab
from xpmir.vocab.encoders import DualTextEncoder


class RandomVocab(Vocab):
    DIMENSION = 7
    MAX_WORDS = 100

    def __init__(self):
        super().__init__()
        self.map = {}
        self.embed = torch.nn.Embedding.from_pretrained(
            torch.randn(RandomVocab.MAX_WORDS, RandomVocab.DIMENSION)
        )

    def dim(self) -> int:
        return RandomVocab.DIMENSION

    @property
    def pad_tokenid(self) -> int:
        return 0

    def tok2id(self, tok: str) -> int:
        try:
            return self.map[tok]
        except KeyError:
            tokid = len(self.map)
            self.map[tok] = tokid
            return tokid

    def forward(self, tok_texts: TokenizedTexts):
        return self.embed(tok_texts.ids)

    def static(self) -> bool:
        return False


class CustomIndex(Index):
    @property
    def documentcount(self):
        return 50

    def term_df(self, term: str):
        return 1


# ---
# --- Model factories
# ---


def drmm():
    """Drmm factory"""
    from xpmir.neural.drmm import Drmm

    return Drmm(vocab=RandomVocab(), index=CustomIndex()).instance()


def colbert():
    """Colbert model factory"""
    from xpmir.neural.colbert import Colbert

    return Colbert(
        vocab=RandomVocab(), masktoken=False, doctoken=False, querytoken=False
    ).instance()


class DummyDualTextEncoder(DualTextEncoder):
    @property
    def dimension(self) -> int:
        return 13

    def static(self):
        return False

    def forward(self, texts: List[Tuple[str, str]]):
        return torch.randn(len(texts), 13)


def joint():
    """Joint classifier factory"""
    from xpmir.neural.jointclassifier import JointClassifier

    return JointClassifier(encoder=DummyDualTextEncoder()).instance()


# ---
# --- Input factory
# ---


def pointwise():
    # Pointwise inputs
    inputs = PointwiseRecords()
    inputs.add(PointwiseRecord(Query("purple cat"), "d1", "the purple car", 1, 1))
    return inputs


def pairwise():
    inputs = PairwiseRecords()
    inputs.add(
        PairwiseRecord(
            Query("purple cat"),
            Document("1", "the cat sat on the mat", 1),
            Document("2", "the purple car", 1),
        )
    )
    return inputs


def cartesian():
    inputs = ProductRecords()
    inputs.addDocuments(
        Document("1", "the cat sat on the mat", 1),
        Document("2", "the purple car", 1),
    )

    inputs.addQueries(Query("purple cat"), Query("blue tiger"))

    return inputs


@pytest.mark.parametrize("modelfactory", [drmm, colbert, joint])
@pytest.mark.parametrize("inputfactory", [pointwise, pairwise, cartesian])
def test_model_forward(modelfactory, inputfactory):
    model = modelfactory()
    random = Random().instance().state
    model.initialize(random)

    inputs = inputfactory()

    logging.debug("%s", model(inputs))
