from xpmir.letor.trainers.pointwise import PointwiseTrainer


def test_pointwise():
    sampler = PointwiseSampler()
    PointwiseTrainer(sampler=sampler).instance()
    assert False
