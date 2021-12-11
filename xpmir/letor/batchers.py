from typing import (
    Generic,
    Protocol,
    Callable,
    Any,
    Iterator,
    SupportsIndex,
    Tuple,
    TypeVar,
    Union,
    overload,
)
from experimaestro import Config
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from xpmir.utils import easylog

logger = easylog()

RT = TypeVar("RT")
T = TypeVar("T")
ARGS = TypeVar("ARGS")
KWARGS = TypeVar("KWARGS")


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
        self, batch: Iterator[Sliceable[T]], *args: ARGS, **kwargs: KWARGS
    ) -> RT:
        ...


class Processor(Protocol, Generic[T, ARGS, KWARGS]):
    def __call__(self, batch: Sliceable[T], *args: ARGS, **kwargs: KWARGS) -> None:
        ...


class BatcherWorker:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def process_withreplay(
        self,
        batch: Sliceable[T],
        process: IterativeProcessor[T, RT, ARGS, KWARGS],
        *args: ARGS,
        **kwargs: KWARGS
    ) -> RT:
        # Just do nothing
        return process(iter([batch]), *args, **kwargs)

    def process(
        self,
        batch: Sliceable[T],
        process: Processor[T, ARGS, KWARGS],
        *args: ARGS,
        **kwargs: KWARGS
    ) -> None:
        process(batch, *args, **kwargs)


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
        ix = 0
        while ix < batch_size:
            yield slice(ix, ix + self.batch_size)
            ix += self.batch_size

    def iter(self, batch: Sliceable[T]) -> Iterator[Sliceable[T]]:
        for range in self.get_ranges(len(batch)):
            yield batch[range]

    def process_withreplay(
        self,
        batch: Sliceable[T],
        process: IterativeProcessor[T, RT, ARGS, KWARGS],
        *args: ARGS,
        **kwargs: KWARGS
    ) -> RT:
        while True:
            flag, rt = self._run(lambda: process(self.iter(batch), *args, **kwargs))
            if flag:
                return rt

    def process(
        self,
        batch: Sliceable[T],
        process: Processor[T, ARGS, KWARGS],
        *args: ARGS,
        **kwargs: KWARGS
    ) -> None:
        ix = 0
        while ix < len(batch):
            s = slice(ix, ix + self.batch_size)
            flag, rt = self._run(lambda: process(batch[s], *args, **kwargs))
            if flag:
                ix += self.batch_size

    def _run(self, process: Callable[[], RT]) -> Tuple[bool, Union[RT, None]]:
        from pytorch_lightning.utilities.memory import is_oom_error

        try:
            # Perform a process
            return True, process()
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
                return False, None
            else:
                # Other exception
                raise


class PowerAdaptativeBatcher(Batcher):
    """Starts with the provided batch size, and then divides in 2, 3, etc.
    until there is no more OOM
    """

    def initialize(self, batch_size: int) -> PowerAdaptativeBatcherWorker:
        return PowerAdaptativeBatcherWorker(batch_size)
