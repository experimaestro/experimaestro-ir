"""SamplerAdapter: wraps a sampler with processors that transform its output.

Processors run in batches for efficiency (e.g., batch store lookups).
The result is a new Sampler with potentially different doc/query types.
"""

from typing import TypeVar, List, Iterator, Any, Dict

from experimaestro import Param, field
from xpm_torch.base import Sampler
from xpm_torch.datasets import ShardedIterableDataset
from xpmir.letor.processors import RecordsProcessor

SampleT = TypeVar("SampleT")
DocT = TypeVar("DocT")
QueryT = TypeVar("QueryT")
DocT2 = TypeVar("DocT2")
QueryT2 = TypeVar("QueryT2")


class BufferedProcessingDataset(ShardedIterableDataset):
    """Buffers records from inner dataset, batch-processes, yields one-by-one."""

    def __init__(
        self,
        inner: ShardedIterableDataset,
        processors: List[RecordsProcessor],
        buffer_size: int,
    ):
        super().__init__()
        self.inner = inner
        self.processors = processors
        self.buffer_size = buffer_size

    def _flush(self, records):
        for proc in self.processors:
            records = proc.process_batch(records)
        return records

    def iter_shard(self, shard_id: int, num_shards: int) -> Iterator:
        buffer = []
        for record in self.inner.iter_shard(shard_id, num_shards):
            buffer.append(record)
            if len(buffer) >= self.buffer_size:
                yield from self._flush(buffer)
                buffer = []
        if buffer:
            yield from self._flush(buffer)

    def state_dict(self) -> Dict[str, Any]:
        return self.inner.state_dict()

    def load_state_dict(self, state: Dict[str, Any]):
        self.inner.load_state_dict(state)


class SamplerAdapter(Sampler):
    """Wraps a sampler with processors that transform its output.

    Processors run in batches for efficiency (e.g., batch store lookups).
    The result is a new Sampler with potentially different doc/query types.
    """

    sampler: Param[Sampler]
    processors: Param[List[RecordsProcessor]]
    buffer_size: Param[int] = field(default=64)

    def initialize(self, random):
        super().initialize(random)
        self.sampler.initialize(random)

    def as_dataset(self) -> ShardedIterableDataset:
        return BufferedProcessingDataset(
            self.sampler.as_dataset(), self.processors, self.buffer_size
        )
