import logging
from typing import List, Tuple
import itertools
import pytest
import torch
from collections import defaultdict
from datamaestro_text.data.ir import TextItem, SimpleTextItem, IDTextRecord
from xpmir.neural.dual import CosineDense, DotDense
from xpmir.letor.records import (
    PairwiseRecord,
    PairwiseRecords,
    PointwiseRecord,
    PointwiseRecords,
    ProductRecords,
)
from xpmir.text.tokenizers import Tokenizer
from xpmir.text.encoders import (
    TokenizedTextEncoderBase,
    DualTextEncoder,
    TokensRepresentationOutput,
    TokenizerOptions,
    RepresentationOutput,
)
from xpmir.text.adapters import MeanTextEncoder


class MyTokenizer(Tokenizer):
    def __post_init__(self):
        super().__post_init__()
        self.map = {}
        self._dummy_params = torch.nn.Parameter(torch.Tensor())

    def tok2id(self, tok: str) -> int:
        try:
            return self.map[tok]
        except KeyError:
            tokid = len(self.map)
            self.map[tok] = tokid
            return tokid


class RandomTokensEncoder(TokenizedTextEncoderBase[IDTextRecord, TokensRepresentationOutput]):
    DIMENSION = 7
    MAX_WORDS = 100

    def __initialize__(self):
        super().__initialize__()
        self.embed = torch.nn.Embedding.from_pretrained(
            torch.randn(RandomTokensEncoder.MAX_WORDS, RandomTokensEncoder.DIMENSION)
        )
        self.tokenizer = MyTokenizer.C().instance()

    @property
    def dimension(self) -> int:
        return RandomTokensEncoder.DIMENSION

    @property
    def pad_tokenid(self) -> int:
        return 0

    def forward(self, records: List[IDTextRecord], options=None):
        options = options or TokenizerOptions()
        tok_texts = self.tokenizer.batch_tokenize(
            [record["text_item"].text for record in records],
            maxlen=options.max_length,
            mask=True,
        )
        return TokensRepresentationOutput(self.embed(tok_texts.ids), tok_texts)

    def static(self) -> bool:
        return False


# ---
# --- Model factories
# ---

modelfactories = []


def registermodel(method):
    modelfactories.append(
        pytest.param(method, marks=pytest.mark.dependency(name=f"model-{method}"))
    )
    return method


@registermodel
def dotdense():
    """Dense model factory"""
    return DotDense.C(
        encoder=MeanTextEncoder.C(encoder=RandomTokensEncoder.C()),
        query_encoder=MeanTextEncoder.C(encoder=RandomTokensEncoder.C()),
    ).instance()


@registermodel
def cosinedense():
    """Cosine model factory"""
    return CosineDense.C(
        encoder=MeanTextEncoder.C(encoder=RandomTokensEncoder.C()),
        query_encoder=MeanTextEncoder.C(encoder=RandomTokensEncoder.C()),
    ).instance()


class DummyDualTextEncoder(DualTextEncoder):
    def __initialize__(self):
        super().__initialize__()
        self.cache = defaultdict(lambda: torch.randn(1, 13))

    @property
    def dimension(self) -> int:
        return 13

    def static(self):
        return False

    def forward(self, texts: List[Tuple[str, str]]):
        return RepresentationOutput(torch.cat([self.cache[text] for text in texts]))


# ---
# --- Input factory
# ---

QUERIES = [
    {"text_item": SimpleTextItem("purple cat")},
    {"text_item": SimpleTextItem("yellow house")},
]
DOCUMENTS = [
    {"id": "1", "text_item": SimpleTextItem("the cat sat on the mat")},
    {"id": "2", "text_item": SimpleTextItem("the purple car")},
    {"id": "3", "text_item": SimpleTextItem("my little dog")},
    {"id": "4", "text_item": SimpleTextItem("the truck was on track")},
]


def pointwise():
    # Pointwise inputs
    inputs = PointwiseRecords()

    # Implicit order (Q0, D0) (Q1, D0) (Q0, D1) (Q1, D1)
    inputs.add(PointwiseRecord(QUERIES[0], DOCUMENTS[0], 0.0))
    inputs.add(PointwiseRecord(QUERIES[0], DOCUMENTS[0], 0.0))
    inputs.add(PointwiseRecord(QUERIES[1], DOCUMENTS[2], 0.0))
    inputs.add(PointwiseRecord(QUERIES[1], DOCUMENTS[1], 0.0))
    return inputs


def pairwise():
    # Implicit order (Q0, D0) (Q1, D0) (Q0, D1) (Q1, D1)
    inputs = PairwiseRecords()
    inputs.add(PairwiseRecord(QUERIES[0], DOCUMENTS[0], DOCUMENTS[1]))
    inputs.add(PairwiseRecord(QUERIES[1], DOCUMENTS[2], DOCUMENTS[3]))
    return inputs


def product():
    # Implicit order (Q0, D0) (Q0, D1) (Q1, D0)
    inputs = ProductRecords()
    inputs.add_topics(QUERIES[0], QUERIES[1])
    inputs.add_documents(DOCUMENTS[0], DOCUMENTS[1])

    return inputs


inputfactories = [pointwise, pairwise, product]


@pytest.mark.parametrize("modelfactory", modelfactories)
@pytest.mark.parametrize("inputfactory", inputfactories)
@pytest.mark.dependency()
def test_forward_types(modelfactory, inputfactory):
    """Test that each record type is handled"""
    model = modelfactory()
    model.initialize()

    inputs = inputfactory()

    logging.debug("%s", model(inputs, None))


@pytest.mark.parametrize("modelfactory", modelfactories)
@pytest.mark.parametrize(
    "inputfactoriescouple",
    (
        pytest.param((f1, f2), id=f"{f1.__name__}-{f2.__name__}")
        for f1, f2 in itertools.combinations(inputfactories, 2)
    ),
)
def test_forward_consistency(modelfactory, inputfactoriescouple):
    """Test that outputs are consistent between the different records types"""
    model = modelfactory()
    model.initialize()

    outputs = []
    maps = []
    with torch.no_grad():
        for f in inputfactoriescouple:
            input = f()
            outputs.append(model(input, None))
            maps.append(
                {
                    (qr["text_item"].text, dr["text_item"].text): ix
                    for ix, (qr, dr) in enumerate(zip(input.queries, input.documents))
                }
            )

    inter = set(maps[0].keys() & maps[1].keys())
    assert len(inter) > 0, "No common query/document pair"
    for key in inter:
        s1 = outputs[0][maps[0][key]].item()
        s2 = outputs[1][maps[1][key]].item()
        assert s1 == pytest.approx(s2, abs=1e-6), (
            f"{s1} different from {s2} in {outputs[0]}, {outputs[1]}"
        )
