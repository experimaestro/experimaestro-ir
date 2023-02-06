import numpy as np
from typing import Callable, Dict, Generic, Iterable, Iterator, Protocol, TypeVar
from xpmir.utils.utils import easylog
from abc import abstractmethod

logger = easylog()

T = TypeVar("T")

# --- Utility classes

T = TypeVar("T")
U = TypeVar("U")
ItType = TypeVar("ItType", covariant=True)


class SerializableIterator(Iterator[ItType], Protocol[ItType]):
    """An iterator that can be serialized through state dictionaries.

    This is used when saving the sampler state
    """

    def state_dict(self) -> Dict:
        ...

    def load_state_dict(self, state):
        ...


class SerializableIteratorAdapter(Iterable[U], Generic[T, U]):
    """Adapts a serializable iterator with a transformation function"""

    def __init__(
        self,
        main: SerializableIterator[T],
        generator: Callable[[SerializableIterator[T]], Iterator[U]],
    ):
        self.generator = generator
        self.main = main
        self.iter = generator(main)

    def load_state_dict(self, state):
        self.main.load_state_dict(state)
        self.iter = self.generator(self.main)

    def state_dict(self):
        return self.main.state_dict()

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)


class GenericSerializableIterator(SerializableIterator[T]):
    def __init__(self, iterator: Iterator[T]):
        self.iterator = iterator
        self.state = None

    @abstractmethod
    def state_dict(self):
        """Generate the current state dictionary"""
        ...

    @abstractmethod
    def restore_state(self, state):
        """Restore the iterator"""
        ...

    def load_state_dict(self, state):
        self.state = state

    def __next__(self):
        # Nature of the documents
        if self.state is not None:
            self.restore_state(self.state)
            self.state = None

        # And now go ahead
        return self.next()


class RandomSerializableIterator(SerializableIterator[T]):
    """A serializable iterator based on a random seed"""

    def __init__(
        self,
        random: np.random.RandomState,
        generator: Callable[[np.random.RandomState], Iterator[T]],
    ):
        """Creates a new iterator based on a random generator

        Args:
            random (np.random.RandomState): The initial random state

            generator (Callable[[np.random.RandomState], Iterator[T]]): Generate
            a new iterator from a random seed
        """
        self.random = random
        self.generator = generator
        self.iter = generator(random)

    def load_state_dict(self, state):
        self.random.set_state(state["random"])
        self.iter = self.generator(self.random)

    def state_dict(self):
        return {"random": self.random.get_state()}

    def __next__(self):
        return next(self.iter)


class SkippingIterator(GenericSerializableIterator[T]):
    """An iterator that skips the first entries and can output its state

    When serialized (i.e. checkpointing), the iterator saves the current
    position. This can be used when deserialized, to get back to the same
    (checkpointed) position.
    """

    position: int
    """The current position (in number of items) of the iterator"""

    def __init__(self, iterator: Iterator[T]):
        super().__init__(iterator)
        self.position = 0

    def state_dict(self):
        return {"count": self.position}

    def restore_state(self, state):
        count = state["count"]
        logger.info("Skipping %d records to match state (sampler)", count)

        assert count >= self.position, "Cannot iterate backwards"
        for _ in range(count - self.position):
            next(self.iterator)
        self.position = count

    def next(self):
        self.position += 1
        return next(self.iterator)
