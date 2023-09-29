import numpy as np
from xpmir.utils.iter import (
    RandomSerializableIterator,
    SkippingIterator,
    MultiprocessSerializableIterator,
)


def test_iter_skipping_iterator():
    a = list(range(10))

    iterator = SkippingIterator(iter(a))
    for _ in range(5):
        next(iterator)

    state = iterator.state_dict()
    x = next(iterator)

    iterator = SkippingIterator(iter(a))
    iterator.load_state_dict(state)
    assert next(iterator) == x


def test_iter_random_iterator():
    a = list(range(10))

    def create_iter(state: np.random.RandomState):
        while True:
            ix = state.randint(len(a))
            yield a[ix]

    rs = np.random.RandomState()
    iterator = RandomSerializableIterator(rs, create_iter)
    for _ in range(5):
        next(iterator)

    state = iterator.state_dict()
    x = next(iterator)

    iterator = RandomSerializableIterator(rs, create_iter)
    iterator.load_state_dict(state)
    assert next(iterator) == x


def test_iter_mp_iterator():
    """Test the multiprocess iterator"""
    a = list(range(10))

    iterator = MultiprocessSerializableIterator(SkippingIterator(iter(a)))
    for _ in range(5):
        next(iterator)

    state = iterator.state_dict()
    x = next(iterator)

    iterator = MultiprocessSerializableIterator(SkippingIterator(iter(a)))
    iterator.load_state_dict(state)
    assert next(iterator) == x
