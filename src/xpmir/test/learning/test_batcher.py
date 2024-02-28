from typing import Iterator, List
from xpmir.learning.batchers import PowerAdaptativeBatcher


def add(x: int, y: int):
    return x + y


class Processor:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, batch: List[int], results: List[int], *args):
        for x in batch:
            results.append(self.fn(x, *args))


class IteratorProcessor:
    def __init__(self, fn):
        self.fn = fn
        self.calls = 0

    def __call__(self, batches: Iterator[List[int]], count: int, *args):
        self.calls += 1
        results = []
        for batch in batches:
            count -= 1
            for x in batch:
                results.append(self.fn(x, *args))
        assert count == 0
        return results


def apply(fn, data: List[int], *args):
    results: List[int] = []
    for x in data:
        results.append(fn(x, *args))
    return results


class FailingAdd:
    def __init__(self, fail: int):
        self.fail = fail

    def __call__(self, x: int, y: int):
        if x == self.fail:
            self.fail = None
            # Simulates a CUDA out-of-memory error
            raise RuntimeError("CUBLAS_STATUS_ALLOC_FAILED")
        return x + y


def test_poweradaptativebatcher():
    batcher: PowerAdaptativeBatcher = PowerAdaptativeBatcher().instance()
    worker = batcher.initialize(3)

    data = list(range(7))
    args = [4]
    expected = apply(add, data, *args)

    # Test without trouble
    observed = []
    worker.process(data, Processor(add), observed, *args)
    assert observed == expected

    # Test with OOM handling
    worker = batcher.initialize(3)
    failing_add = FailingAdd(3)
    observed = []
    worker.process(data, Processor(failing_add), observed, *args)
    assert failing_add.fail is None
    assert observed == expected

    # Test with OOM and restart
    worker = batcher.initialize(3)
    failing_add = FailingAdd(3)
    processor = IteratorProcessor(failing_add)
    observed = worker.process_withreplay(data, processor, *args)
    assert failing_add.fail is None
    assert processor.calls == 2
    assert observed == expected
