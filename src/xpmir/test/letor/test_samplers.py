from typing import Iterator, Tuple
from datamaestro_text.data.ir.base import GenericTopic, GenericDocument
from xpmir.letor.samplers import TrainingTriplets, TripletBasedSampler

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
