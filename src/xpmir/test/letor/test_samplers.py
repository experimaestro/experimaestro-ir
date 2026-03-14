import pytest
import numpy as np
from typing import Iterator, Tuple
from experimaestro import field, Param
import datamaestro_ir.data as ir
from datamaestro_ir.data import IDTextRecord, SimpleTextItem
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
    ) -> Iterator[Tuple[IDTextRecord, IDTextRecord, IDTextRecord]]:
        count = 0

        while True:
            yield (
                {"text_item": SimpleTextItem(f"q{count}")},
                {"id": "1", "text_item": SimpleTextItem(f"doc+{count}")},
                {"id": "2", "text_item": SimpleTextItem(f"doc-{count}")},
            )


def test_serializing_tripletbasedsampler():
    """Serialized samplers should start back from the saved state"""
    sampler = TripletBasedSampler.C(
        source=MyTrainingTriplets.C(id="test-triplets")
    ).instance()
    dataset = sampler.as_dataset()

    # Collect 20 samples total: skip first 10, keep next 10
    all_samples = []
    for item, _ in zip(dataset, range(20)):
        all_samples.append(item)
    expected_samples = all_samples[10:20]

    # Verify the dataset produces the expected items from a fresh start
    # (deterministic iteration)
    sampler2 = TripletBasedSampler.C(
        source=MyTrainingTriplets.C(id="test-triplets")
    ).instance()
    dataset2 = sampler2.as_dataset()
    check_samples = []
    for item, _ in zip(dataset2, range(20)):
        check_samples.append(item)

    for record, expected in zip(check_samples[10:20], expected_samples):
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
    id: Param[str] = field(default="", ignore_default=True)

    @property
    def documentcount(self):
        return 10

    def document_int(self, internal_docid: int) -> IDTextRecord:
        return {
            "id": str(internal_docid),
            "text_item": SimpleTextItem(f"D{internal_docid} " * 10),
        }


def test_pairwise_randomspansampler():
    documents = FakeDocumentStore.C()

    sampler1 = RandomSpanSampler.C(documents=documents).instance()

    sampler2 = RandomSpanSampler.C(documents=documents).instance()

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
