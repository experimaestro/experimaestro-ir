import numpy as np
import atexit
from abc import ABC, abstractmethod
from queue import Full, Empty
import torch.multiprocessing as mp
from typing import (
    Generic,
    Callable,
    Dict,
    Tuple,
    List,
    Iterable,
    Iterator,
    Protocol,
    TypeVar,
    Any,
    TypedDict,
)
from xpmir.utils.utils import easylog
import logging

logger = easylog()

# --- Utility classes

State = TypeVar("State")
T = TypeVar("T")
U = TypeVar("U")


class iterable_of(Generic[T]):
    def __init__(self, factory: Callable[[], Iterator[T]]):
        self.factory = factory

    def __iter__(self):
        return self.factory()


class SerializableIterator(Iterator[T], Generic[T, State]):
    """An iterator that can be serialized through state dictionaries.

    This is used when saving the sampler state
    """

    @abstractmethod
    def state_dict(self) -> State:
        ...

    @abstractmethod
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


class BatchIteratorAdapter(SerializableIterator[List[T], State]):
    """Adapts a serializable iterator into a batchwise serializable iterator"""

    def __init__(self, iterator: SerializableIterator[T, State], size: int):
        self.iterator = iterator
        self.size = size

    def state_dict(self):
        return self.iterator.state_dict()

    def load_state_dict(self, state):
        self.iterator.load_state_dict(state)

    def __iter__(self):
        return self

    def __next__(self) -> List[T]:
        batch = []
        for _, record in zip(range(self.size), self.iterator):
            batch.append(record)
        return batch


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


class RandomizedSerializableIteratorState(TypedDict):
    random: Dict[str, Any]
    state: Any


class RandomStateSerializableIterator(SerializableIterator[T, State], ABC):
    @abstractmethod
    def set_random(self, random: np.random.RandomState):
        ...


class RandomStateSerializableAdaptor(RandomStateSerializableIterator[T, State], ABC):
    """Adapter for random state-biased iterator"""

    def __init__(self, iterator: SerializableIterator[T, State]):
        self.random = None
        self.iterator = iterator

    def set_random(self, random: np.random.RandomState):
        self.random = random

    def load_state_dict(self, state: State):
        return self.iterator.load_state_dict(state)

    def state_dict(self) -> State:
        return self.iterator.state_dict()


class RandomizedSerializableIterator(
    RandomSerializableIterator[T, RandomizedSerializableIteratorState[State]],
    Generic[T, State],
):
    """Serializable iterator with a random state"""

    def __init__(
        self, random: np.random.RandomState, iterator: RandomStateSerializableIterator
    ):
        """Creates a new iterator based on a random generator

        Args:
            random (np.random.RandomState): The initial random state

            generator (Callable[[np.random.RandomState], Iterator[T]]): Generate
            a new iterator from a random seed
        """
        self.random = random
        self.iterator = iterator
        iterator.set_random(self.random)

    def load_state_dict(self, state: RandomizedSerializableIteratorState[State]):
        self.random.set_state(state["random"])
        self.iterator.set_random(self.random)
        self.iterator.load_state_dict(state["state"])

    def state_dict(self) -> RandomizedSerializableIteratorState[State]:
        return {"random": self.random.get_state(), "state": self.iterator.state_dict()}

    def __next__(self):
        return next(self.iterator)


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

    def next(self) -> T:
        self.position += 1
        return next(self.iterator)

    @staticmethod
    def make_serializable(iterator):
        if not isinstance(iterator, SerializableIterator):
            logging.info("Wrapping iterator into a skipping iterator")
            return SkippingIterator(iterator)

        return iterator


class InfiniteSkippingIterator(SkippingIterator[T, SkippingIteratorState]):
    """Subclass of the SkippingIterator that loops an infinite number of times"""

    def __init__(self, iterable: Iterable[T]):
        super().__init__(iter(iterable))
        self.iterable = iterable

    def next(self) -> T:
        try:
            return super().next()
        except StopIteration:
            self.iterator = iter(self.iterable)
            self.position = 1
            return next(self.iterator)


class StopIterationClass:
    pass


STOP_ITERATION = StopIterationClass()


def mp_iterate(iterator, queue: mp.Queue, event: mp.Event):
    try:
        while not event.is_set():
            value = next(iterator)

            while True:
                try:
                    queue.put(value, timeout=1)
                    break
                except Full:
                    if event.is_set():
                        logger.warning("Stopping as requested by the main process")
                        queue.close()
                        break

    except StopIteration:
        logger.info("Signaling that the iterator has finished")
        queue.put(STOP_ITERATION)
    except Exception as e:
        logger.exception("Exception while iterating")
        queue.put(e)

    logger.info("End of multi-process iterator")
    queue.close()


class QueueBasedMultiprocessIterator(Iterator[T]):
    """This Queue-based iterator can be pickled when a new process is spawn"""

    def __init__(self, queue: "mp.Queue[T]", stop_process: mp.Event):
        self.queue = queue
        self.stop_process = stop_process
        self.stop_iteration = mp.Event()

    def __next__(self):
        # Get the next element
        while True:
            try:
                element = self.queue.get(timeout=1)
                break
            except Empty:
                if self.stop_iteration.is_set():
                    self.stop_process.set()
                    raise StopIteration()

        # Last element
        if isinstance(element, StopIterationClass):
            # Just in case
            self.stop_process.set()
            self.stop_iteration.set()
            raise StopIteration()

        # An exception occurred
        elif isinstance(element, Exception):
            self.stop_iteration.set()
            self.stop_process.set()
            raise RuntimeError("Error in iterator process") from element

        return element


class MultiprocessIterator(Iterator[T]):
    def __init__(self, iterator: Iterator[T], maxsize=100):
        self.process = None
        self.maxsize = maxsize
        self.iterator = iterator
        self.stop_process = mp.Event()
        self.mp_iterator = None

    def start(self):
        """Start the iterator process"""
        if self.process is None:
            self.queue = mp.Queue(self.maxsize)
            self.process = mp.Process(
                target=mp_iterate,
                args=(self.iterator, self.queue, self.stop_process),
                daemon=True,
            )

            # Start the process
            self.process.start()
            self.mp_iterator = QueueBasedMultiprocessIterator(
                self.queue, self.stop_process
            )

            atexit.register(self.close)
        return self

    def close(self):
        if self.mp_iterator:
            atexit.unregister(self.close)
            self.stop_process.set()
            self.mp_iterator = None
            logging.info("Signaled the mp_iterator to quit")

    def detach(self):
        """Produces an iterator only based on the multiprocess queue (useful
        when using torch mp.spawn)"""
        self.start()
        return self.mp_iterator

    def __next__(self):
        # Start a process if needed
        self.start()
        try:
            return next(self.mp_iterator)
        except StopIteration:
            atexit.unregister(self.close)


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
