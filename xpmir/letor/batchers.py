from typing import Callable, Generic, Iterator, List, Tuple, TypeVar
from experimaestro import Config
from pytorch_lightning.utilities.memory import garbage_collection_cuda
from xpmir.utils import easylog

logger = easylog()

RT = TypeVar("RT")


class BatcherWorker(Generic[RT]):
    def __init__(self, batch_size: int, process: Callable[[Iterator], RT]):
        self.process = process
        self.batch_size = batch_size

    def __call__(self, batch, *args, **kwargs) -> RT:
        # Just do nothing
        return self.process(iter([batch]), *args, **kwargs)


class Batcher(Config):
    """Responsible for micro-batching when the batch does not fit in memory

    The base class just does nothing (no adaptation)
    """

    def initialize(
        self, batch_size: int, process: Callable[[Iterator], RT]
    ) -> BatcherWorker[RT]:
        logger.info("Using a simple batcher with batch size %d", batch_size)
        return BatcherWorker(batch_size, process)


class PowerAdaptativeBatcherWorker(BatcherWorker[RT]):
    def __init__(self, batch_size: int, process: Callable[[Iterator], RT]):
        super().__init__(batch_size, process)
        self.max_batch_size = batch_size
        self.current_divider = 1
        logger.info("Adaptative batcher: initial batch size is %d", self.batch_size)

    def get_ranges(self, batch_size):
        ix = 0
        while ix < batch_size:
            yield slice(ix, ix + self.batch_size)
            ix += self.batch_size

    def iter(self, batch):
        for range in self.get_ranges(len(batch)):
            yield batch[range]

    def __call__(self, batch, *args, **kwargs):
        from pytorch_lightning.utilities.memory import is_oom_error

        while True:
            try:
                # Perform a process
                return self.process(self.iter(batch), *args, **kwargs)
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
                else:
                    # Other exception
                    raise


class PowerAdaptativeBatcher(Batcher):
    """Starts with the provided batch size, and then divides in 2, 3, etc.
    until there is no more OOM
    """

    def initialize(
        self, batch_size: int, process: Callable[[Iterator], RT]
    ) -> PowerAdaptativeBatcherWorker[RT]:
        return PowerAdaptativeBatcherWorker(batch_size, process)
