"""Index for sparse models"""

import os
from queue import Full, Empty
import torch.multiprocessing as mp
from typing import Any, Generic, TypeVar
from xpmir.utils.logging import easylog

logger = easylog()

T = TypeVar("T")


def available_cpus():
    """Returns the number of available CPU cores"""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return mp.cpu_count()


class StoppableQueue(Generic[T]):
    """Queue with a stop event flag"""

    def __init__(self, maxsize: int, stopping_event: mp.Event):
        self.queue = mp.Queue(maxsize)
        self._stopping_event = stopping_event

    def get(self, timeout=1.0):
        item = None
        while True:
            try:
                item = self.queue.get(timeout=timeout)
                break
            except Empty:
                if self._stopping_event.is_set():
                    raise

        return item

    def put(self, item: Any, timeout=1.0):
        while True:
            try:
                self.queue.put(item, timeout=timeout)
                break
            except Full:
                # Try again...
                if self._stopping_event.is_set():
                    raise

    def close(self):
        self.queue.close()

    def stop(self):
        logger.warning("Stopping the iterator")
        self._stopping_event.set()
