from typing import (
    Generic,
    List,
    Protocol,
    Callable,
    Iterator,
    TypeVar,
    Union,
    overload,
)
from experimaestro import Config
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from xpmir.utils.utils import easylog

logger = easylog()

RT = TypeVar("RT")
T = TypeVar("T")
ARGS = TypeVar("ARGS")
KWARGS = TypeVar("KWARGS")


class RecoverableOOMError(Exception):
    """Exception raised when a OOM occurs and the batcher
    is taking this into account for the next round (i.e. we
    can reprocess the batch with a higher probability of not
    having an OOM)
    """

    pass


class Sliceable(Protocol, Generic[T]):
    @overload
    def __getitem__(self, slice: slice) -> "Sliceable[T]":
        ...

    @overload
    def __getitem__(self, slice: int) -> "T":
        ...

    def __len__(self) -> int:
        ...


class IterativeProcessor(Protocol, Generic[T, RT, ARGS, KWARGS]):
    def __call__(
        self, batch: Iterator[Sliceable[T]], length: int, *args: ARGS, **kwargs: KWARGS
    ) -> RT:
        ...


class Processor(Protocol, Generic[T, ARGS, KWARGS]):
    def __call__(self, batch: Sliceable[T], *args: ARGS, **kwargs: KWARGS) -> None:
        ...


class Reducer(Protocol, Generic[T, RT, ARGS, KWARGS]):
    def __call__(
        self, batch: Sliceable[T], value: RT, *args: ARGS, **kwargs: KWARGS
    ) -> RT:
        ...


class BatcherWorker:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def process_withreplay(
        self,
        batch: Sliceable[T],
        process: IterativeProcessor[T, RT, ARGS, KWARGS],
        *args: ARGS,
        **kwargs: KWARGS,
    ) -> RT:
        """Process a batch with replay

        Replay = if an error occurs, the full batch is re-processed
        """
        return process(iter([batch]), 1, *args, **kwargs)

    def process(
        self,
        batch: Sliceable[T],
        process: Processor[T, ARGS, KWARGS],
        *args: ARGS,
        raise_oom=False,
        **kwargs: KWARGS,
    ) -> None:
        """Process a batch

        If a recoverable OOM error occurs, the processing continues but do not
        reprocess previously processed samples.

        Arguments:

            raise_oom: Raise an OOM exception when an OOM is recoverable
        """
        process(batch, *args, **kwargs)

    def reduce(
        self,
        batch: Sliceable[T],
        reducer: Reducer[T, RT, ARGS, KWARGS],
        initialvalue: RT,
        *args: ARGS,
        raise_oom=False,
        **kwargs: KWARGS,
    ) -> RT:
        """Attributes:

        Arguments:
            batch: The data to process

            reducer: The reducer function, whose two first arguments are a slice
            of T and the reduced value, and that returns a new value

            raise_oom: Raise an OOM exception when an OOM is recoverable instead
            of continuing
        """
        return reducer(batch, initialvalue, *args, **kwargs)


def is_cublas_alloc_failed(exception):
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUBLAS_STATUS_ALLOC_FAILED" in exception.args[0]
    )


def is_oom_error(exception):
    """Detect CUDA out-of-memory errors"""
    from pytorch_lightning.utilities.memory import is_oom_error as legacy_check

    if legacy_check(exception):
        return True

    return is_cublas_alloc_failed(exception)


class Batcher(Config):
    """Responsible for micro-batching when the batch does not fit in memory

    The base class just does nothing (no adaptation)
    """

    def initialize(self, batch_size: int) -> BatcherWorker:
        logger.info("Using a simple batcher with batch size %d", batch_size)
        return BatcherWorker(batch_size)


class PowerAdaptativeBatcherWorker(BatcherWorker):
    def __init__(self, batch_size: int):
        super().__init__(batch_size)
        self.max_batch_size = batch_size
        self.current_divider = 1
        logger.info("Adaptative batcher: initial batch size is %d", self.batch_size)

    def get_ranges(self, batch_size):
        ranges = []
        ix = 0
        while ix < batch_size:
            ranges.append(slice(ix, ix + self.batch_size))
            ix += self.batch_size
        return ranges

    def iter(self, batch: Sliceable[T], ranges: List[slice]) -> Iterator[Sliceable[T]]:
        for range in ranges:
            yield batch[range]

    def process_withreplay(
        self,
        batch: Sliceable[T],
        process: IterativeProcessor[T, RT, ARGS, KWARGS],
        *args: ARGS,
        **kwargs: KWARGS,
    ) -> RT:
        while True:
            ranges = self.get_ranges(len(batch))
            rt = self._run(
                lambda: process(self.iter(batch, ranges), len(ranges), *args, **kwargs)
            )
            if not isinstance(rt, RecoverableOOMError):
                return rt

    def process(
        self,
        batch: Sliceable[T],
        process: Processor[T, ARGS, KWARGS],
        *args: ARGS,
        raise_oom=False,
        **kwargs: KWARGS,
    ) -> None:
        ix = 0
        while ix < len(batch):
            s = slice(ix, ix + self.batch_size)
            rt = self._run(lambda: process(batch[s], *args, **kwargs))
            if not isinstance(rt, RecoverableOOMError):
                ix += self.batch_size
            elif raise_oom:
                raise rt

    def reduce(
        self,
        batch: Sliceable[T],
        reducer: Reducer[T, RT, ARGS, KWARGS],
        value: RT,
        *args: ARGS,
        raise_oom=False,
        **kwargs: KWARGS,
    ) -> RT:
        """
        Reduce a batch using the process function

        Args:
            batch: A batch
            reducer: A function that

        raise_oom: Raise an OOM exception when an OOM is recoverable
        """
        ix = 0
        while ix < len(batch):
            s = slice(ix, ix + self.batch_size)
            rt = self._run(lambda: reducer(batch[s], value, *args, **kwargs))
            if not isinstance(rt, RecoverableOOMError):
                ix += self.batch_size
                value = rt
            elif raise_oom:
                raise RecoverableOOMError()

        return value

    def _run(self, process: Callable[[], RT]) -> Union[RecoverableOOMError, RT]:
        try:
            # Perform a process
            return process()
        except RuntimeError as exception:
            if is_oom_error(exception):
                garbage_collection_cuda()
                while True:
                    self.current_divider += 1
                    new_batchsize = self.max_batch_size // self.current_divider
                    if new_batchsize != self.batch_size:
                        break

                self.batch_size = new_batchsize
                if self.batch_size == 0:
                    logger.error("Cannot decrease batch size below 1")
                    raise
                logger.info(
                    "Adaptative batcher: reducing batch size to %d", self.batch_size
                )
                return RecoverableOOMError(f"Reducing batch size to {self.batch_size}")
            else:
                # Other exception
                raise


class PowerAdaptativeBatcher(Batcher):
    """Starts with the provided batch size, and then divides in 2, 3, etc.
    until there is no more OOM
    """

    def initialize(self, batch_size: int) -> PowerAdaptativeBatcherWorker:
        return PowerAdaptativeBatcherWorker(batch_size)
