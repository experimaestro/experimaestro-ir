import pytest
import numpy as np
from typing import Iterator, Tuple
from experimaestro import Param
import datamaestro_text.data.ir as ir
from datamaestro_text.data.ir.base import GenericTopic, GenericDocument, Document
from xpmir.rankers import Retriever
from xpmir.letor.samplers import (
    TrainingTriplets,
    TripletBasedSampler,
    ModelBasedSampler,
)
from xpmir.documents.samplers import RandomSpanSampler

# ---- Serialization


class MyTrainingTriplets(TrainingTriplets):
    def iter(self) -> Iterator[Tuple[GenericTopic, GenericDocument, GenericDocument]]:
        count = 0

        while True:
            yield GenericTopic(0, f"q{count}"), GenericDocument(
                1, f"doc+{count}"
            ), GenericDocument(2, f"doc-{count}")


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
        assert expected.query.topic.get_text() == record.query.topic.get_text()
        assert (
            expected.positive.document.get_text() == record.positive.document.get_text()
        )
        assert (
            expected.negative.document.get_text() == record.negative.document.get_text()
        )


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

    def count(self):
        return int(100)

    def document_int(self, internal_docid: int) -> Document:
        return GenericDocument(
            str(internal_docid), f"D{internal_docid} D{internal_docid*2}"
        )


def test_pairwise_randomspansampler():
    documents = FakeDocumentStore()

    sampler1 = RandomSpanSampler(documents=documents).instance()

    sampler2 = RandomSpanSampler(documents=documents).instance()

    random = np.random.RandomState()
    sampler1.initialize(random)
    sampler2.initialize(random)
    iter1 = sampler1.pairwise_iter()
    iter2 = sampler2.pairwise_iter()

    for s1, s2, _ in zip(iter1, iter2, range(10)):
        # check that they are the same with same random state
        assert s1.query.get_text() == s2.query.get_text()
        assert s1.positive.get_text() == s2.positive.get_text()
        assert s1.negative.get_text() == s2.negative.get_text()
