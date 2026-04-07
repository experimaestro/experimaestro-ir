"""Index for sparse models"""

import os
from queue import Full, Empty
import torch.multiprocessing as mp
from typing import Any, Generic, TypeVar

import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


def available_cpus():
    """Returns the number of available CPU cores"""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return mp.cpu_count()


def available_memory() -> int:
    """Returns the available memory in bytes.

    Honors SLURM allocation when running inside a SLURM job
    (``SLURM_MEM_PER_NODE`` or ``SLURM_MEM_PER_CPU`` * allocated CPUs).
    Falls back to the OS-reported available memory.
    """
    mem_per_node = os.environ.get("SLURM_MEM_PER_NODE")
    if mem_per_node:
        return int(mem_per_node) * 1024 * 1024

    mem_per_cpu = os.environ.get("SLURM_MEM_PER_CPU")
    if mem_per_cpu:
        cpus = os.environ.get("SLURM_CPUS_ON_NODE") or os.environ.get(
            "SLURM_JOB_CPUS_PER_NODE"
        )
        try:
            n_cpus = int(cpus) if cpus else available_cpus()
        except ValueError:
            n_cpus = available_cpus()
        return int(mem_per_cpu) * n_cpus * 1024 * 1024

    try:
        import psutil

        return psutil.virtual_memory().available
    except ImportError:
        pass

    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, ValueError):
        return 0


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
