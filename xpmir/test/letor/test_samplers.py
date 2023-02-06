from typing import Iterator, Tuple
from xpmir.letor.samplers import TrainingTriplets, TripletBasedSampler

# ---- Serialization


class TestTrainingTriplets(TrainingTriplets):
    def iter(self) -> Iterator[Tuple[str, str, str]]:
        count = 0

        while True:
            yield f"q{count}", f"doc+{count}", f"doc-{count}"


def test_serializing_tripletbasedsampler():
    """Serialized samplers should start back from the saved state"""
    # Collect samples and state after 10 samples
    sampler = TripletBasedSampler(
        source=TestTrainingTriplets(id="test-triplets", ids=False)
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
        source=TestTrainingTriplets(id="test-triplets", ids=False)
    ).instance()
    iter = sampler.pairwise_iter()
    iter.load_state_dict(data)
    for _, record, expected in zip(range(10), iter, samples):
        assert expected.query.text == record.query.text
        assert expected.positive.text == record.positive.text
        assert expected.negative.text == record.negative.text
