import pytest
import numpy as np
from typing import Iterator, Tuple
from experimaestro import Param
import datamaestro_text.data.ir as ir
from xpmir.rankers import Retriever
from xpmir.letor.samplers import (
    TrainingTriplets,
    TripletBasedSampler,
    ModelBasedSampler,
)
from xpmir.documents.samplers import RandomSpanSampler

# ---- Serialization


class MyTrainingTriplets(TrainingTriplets):
    def iter(
        self,
    ) -> Iterator[Tuple[ir.TextRecord, ir.IDTextRecord, ir.IDTextRecord]]:
        count = 0

        while True:
            yield (
                ir.TextRecord(text_item=ir.SimpleTextItem(f"q{count}")),
                ir.IDTextRecord(id=1, text_item=ir.SimpleTextItem(f"doc+{count}")),
                ir.IDTextRecord(id=2, text_item=ir.SimpleTextItem(f"doc-{count}")),
            )


def test_serializing_tripletbasedsampler():
    """Serialized samplers should start back from the saved state"""
    # Collect samples and state after 10 samples
    sampler = TripletBasedSampler(
        source=MyTrainingTriplets(id="test-triplets")
    ).instance()
    iter = sampler.pairwise_iter()

    for _, _ in zip(range(10), iter):
        pass
    data = iter.state_dict()

    samples = []
    for _, record in zip(range(10), sampler.pairwise_iter()):
        samples.append(record)

    # Test
    sampler = TripletBasedSampler(
        source=MyTrainingTriplets(id="test-triplets")
    ).instance()
    iter = sampler.pairwise_iter()
    iter.load_state_dict(data)
    for _, record, expected in zip(range(10), iter, samples):
        assert expected.query["text_item"].text == record.query["text_item"].text
        assert expected.positive["text_item"].text == record.positive["text_item"].text
        assert expected.negative["text_item"].text == record.negative["text_item"].text


class GeneratedDocuments(ir.Documents):
    pass


class GeneratedTopics(ir.Topics):
    pass


class GeneratedAssessments(ir.AdhocAssessments):
    pass


class RandomRetriever(Retriever):
    pass


def adhoc_synthetic_dataset():
    """Creates a random dataset"""
    return ir.Adhoc(
        documents=GeneratedDocuments(),
        topics=GeneratedTopics(),
        assessments=GeneratedAssessments(),
    )


@pytest.mark.skip("not yet done")
def test_modelbasedsampler():
    dataset = adhoc_synthetic_dataset()
    sampler = ModelBasedSampler(
        dataset=dataset, retriever=RandomRetriever(dataset=dataset)
    ).instance()

    for a in sampler._itertopics():
        pass


class FakeDocumentStore(ir.DocumentStore):
    id: Param[str] = ""

    @property
    def documentcount(self):
        return 10

    def document_int(self, internal_docid: int) -> ir.IDTextRecord:
        return ir.IDTextRecord(
            id=str(internal_docid),
            text_item=ir.SimpleTextItem(f"D{internal_docid} " * 10),
        )


def test_pairwise_randomspansampler():
    documents = FakeDocumentStore()

    sampler1 = RandomSpanSampler(documents=documents).instance()

    sampler2 = RandomSpanSampler(documents=documents).instance()

    random1 = np.random.RandomState(seed=0)
    random2 = np.random.RandomState(seed=0)
    sampler1.initialize(random1)
    sampler2.initialize(random2)
    iter1 = sampler1.pairwise_iter()
    iter2 = sampler2.pairwise_iter()

    for s1, s2, _ in zip(iter1, iter2, range(10)):
        # check that they are the same with same random state
        assert s1.query["text_item"].text == s2.query["text_item"].text
        assert s1.positive["text_item"].text == s2.positive["text_item"].text
        assert s1.negative["text_item"].text == s2.negative["text_item"].text
