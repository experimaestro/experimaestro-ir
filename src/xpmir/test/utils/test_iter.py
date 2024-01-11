import numpy as np
from typing import Callable
import logging
from decorator import decorator
from xpmir.utils.iter import (
    RandomSerializableIterator,
    SkippingIterator,
    InfiniteSkippingIterator,
    MultiprocessSerializableIterator,
    RandomizedSerializableIterator,
    SerializableIterator,
    RandomStateSerializableAdaptor,
)


@decorator
def iter_checker(
    factory: Callable[..., SerializableIterator], steps: int = 5, *args, **kw
):
    # Iterate and get state
    iterator = factory(*args, **kw)
    for _ in range(steps):
        next(iterator)

    state = iterator.state_dict()

    # Get the next value
    x = next(iterator)

    # Re-create and test
    iterator = factory()
    iterator.load_state_dict(state)
    assert next(iterator) == x


@iter_checker(steps=5)
def test_iter_skipping_iterator():
    return SkippingIterator(iter(list(range(10))))


@iter_checker(steps=5)
def test_iter_skipping_infinite_iterator():
    return InfiniteSkippingIterator(list(range(3)))


@iter_checker(steps=5)
def test_iter_mp_iterator():
    """Test the multiprocess iterator"""
    return MultiprocessSerializableIterator(SkippingIterator(iter(list(range(10)))))


@iter_checker(steps=5)
def test_iter_randomized_iterator():
    class Iterator(RandomStateSerializableAdaptor[InfiniteSkippingIterator]):
        def __init__(self, n: int):
            base = InfiniteSkippingIterator(list(range(n)))
            super().__init__(base)

        def __next__(self):
            element = next(self.iterator)
            randint = self.random.randint(10)
            logging.debug(
                "---> [%s] %s + %s",
                hash(str(self.random.get_state())),
                element,
                randint,
            )
            return element + randint

    rs = np.random.RandomState()
    return RandomizedSerializableIterator(rs, Iterator(3))


@iter_checker(steps=5)
def test_iter_random_iterator():
    a = list(range(10))

    def create_iter(state: np.random.RandomState):
        while True:
            ix = state.randint(len(a))
            yield a[ix]

    rs = np.random.RandomState()
    return RandomSerializableIterator(rs, create_iter)
