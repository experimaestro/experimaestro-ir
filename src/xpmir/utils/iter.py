import numpy as np
import torch.multiprocessing as mp
from typing import (
    Generic,
    Callable,
    Dict,
    Tuple,
    Iterator,
    Protocol,
    TypeVar,
    Any,
    TypedDict,
)
from xpmir.utils.utils import easylog
from abc import abstractmethod
import logging
import atexit

logger = easylog()

# --- Utility classes

State = TypeVar("State")
T = TypeVar("T")
U = TypeVar("U")


class SerializableIterator(Iterator[T], Generic[T, State]):
    """An iterator that can be serialized through state dictionaries.

    This is used when saving the sampler state
    """

    def state_dict(self) -> State:
        ...

    def load_state_dict(self, state: State):
        ...


class SerializableIteratorAdapter(SerializableIterator[T, State], Generic[T, U, State]):
    """Adapts a serializable iterator with a transformation function based on
    the iterator"""

    def __init__(
        self,
        main: SerializableIterator[T, State],
        generator: Callable[[SerializableIterator[T, State]], Iterator[U]],
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


class SerializableIteratorTransform(
    SerializableIterator[T, State], Generic[T, U, State]
):
    """Adapts a serializable iterator with a transformation function"""

    def __init__(
        self,
        iterator: SerializableIterator[T, State],
        transform: Callable[[T], U],
    ):
        self.transform = transform
        self.iterator = iterator

    def load_state_dict(self, state):
        self.iterator.load_state_dict(state)

    def state_dict(self):
        return self.iterator.state_dict()

    def __iter__(self):
        return self

    def __next__(self):
        return self.transform(next(self.iterator))


class GenericSerializableIterator(SerializableIterator[T, State]):
    def __init__(self, iterator: Iterator[T]):
        self.iterator = iterator
        self.state = None

    @abstractmethod
    def state_dict(self) -> State:
        """Generate the current state dictionary"""
        ...

    @abstractmethod
    def restore_state(self, state: State):
        """Restore the iterator"""
        ...

    def load_state_dict(self, state: State):
        self.state = state

    def __next__(self):
        # Nature of the documents
        if self.state is not None:
            self.restore_state(self.state)
            self.state = None

        # And now go ahead
        return self.next()


class RandomSerializableIterator(SerializableIterator[T, Any]):
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


class SkippingIteratorState(TypedDict):
    """Skipping iterator state"""

    count: int


class SkippingIterator(GenericSerializableIterator[T, SkippingIteratorState]):
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

    def state_dict(self) -> SkippingIteratorState:
        return {"count": self.position}

    def restore_state(self, state: SkippingIteratorState):
        count = state["count"]
        logger.info("Skipping %d records to match state (sampler)", count)

        assert count >= self.position, "Cannot iterate backwards"
        for _ in range(count - self.position):
            next(self.iterator)
        self.position = count

    def next(self):
        self.position += 1
        return next(self.iterator)

    @staticmethod
    def make_serializable(iterator):
        if not isinstance(iterator, SerializableIterator):
            logging.info("Wrapping iterator into a skipping iterator")
            return SkippingIterator(iterator)

        return iterator


class StopIterationClass:
    pass


STOP_ITERATION = StopIterationClass()


def mp_iterate(iterator, queue):
    try:
        while True:
            value = next(iterator)
            queue.put(value)
    except StopIteration:
        queue.put(STOP_ITERATION)
    except Exception as e:
        queue.put(e)


class MultiprocessIterator(Iterator[T]):
    def __init__(self, iterator: Iterator[T], maxsize=100):
        self.process = None
        self.maxsize = maxsize
        self.iterator = iterator

    def __next__(self):
        # (1) Start a process if needed
        if self.process is None:
            self.queue = mp.Queue(self.maxsize)
            self.process = mp.Process(
                target=mp_iterate,
                args=(self.iterator, self.queue),
                daemon=True,
            )

            # Start, and register a kill switch
            self.process.start()
            atexit.register(self.kill_subprocess)

        # Get the next element
        element = self.queue.get()

        # Last element
        if isinstance(element, StopIterationClass):
            atexit.unregister(self.kill_subprocess)
            raise StopIteration()

        # An exception occurred
        if isinstance(element, Exception):
            atexit.unregister(self.kill_subprocess)
            raise RuntimeError("Error in iterator process") from element

        return element

    def kill_subprocess(self):
        # Kills the iterator process
        if self.process and self.process.is_alive():
            logging.debug("Killing iterator process")
            self.process.kill()
            atexit.unregister(self.kill_subprocess)


class StatefullIterator(Iterator[Tuple[T, State]], Protocol[State]):
    """An iterator that iterate over tuples (value, state)"""

    def load_state_dict(self, state: State):
        ...


class StatefullIteratorAdapter(Iterator[T], Generic[T, State]):
    """Adapts a serializable iterator a stateful iterator that iterates over
    (value, state) pairs"""

    def __init__(self, iterator: SerializableIterator[T, State]):
        self.iterator = iterator

    def __next__(self):
        value = next(self.iterator)
        state = self.iterator.state_dict()
        return value, state


class MultiprocessSerializableIterator(
    MultiprocessIterator[T], SerializableIterator[T, State]
):
    """A multi-process adapter for serializable iterators

    This can be used to obtain a multiprocess iterator from a serializable iterator
    """

    def __init__(self, iterator: SerializableIterator[T, State], maxsize=100):
        super().__init__(StatefullIteratorAdapter(iterator), maxsize=maxsize)

    def state_dict(self) -> Dict:
        return self.state

    def load_state_dict(self, state):
        assert self.process is None, "The iterator has already been used"
        self.iterator.iterator.load_state_dict(state)
        self.state = state

    def __next__(self):
        value, self.state = super().__next__()
        return value
