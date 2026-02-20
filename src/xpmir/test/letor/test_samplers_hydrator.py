from functools import cached_property
from experimaestro import Param
from typing import Iterator, Tuple
from datamaestro.record import record_type
import datamaestro_text.data.ir as ir
from xpmir.letor.samplers import (
    TrainingTriplets,
)
from xpmir.datasets.adapters import TextStore

from xpmir.letor.samplers.adapters import BufferedProcessingDataset
from xpmir.letor.processors import StoreHydrator
from xpmir.letor.records import PairwiseRecord
from xpm_torch.datasets import ShardedIterableDataset


class TripletIterator(TrainingTriplets):
    def iter(
        self,
    ) -> Iterator[Tuple[ir.TopicRecord, ir.DocumentRecord, ir.DocumentRecord]]:
        count = 0

        while True:
            yield (
                ir.create_record(id=str(count)),
                ir.create_record(id=str(2 * count)),
                ir.create_record(id=str(2 * count + 1)),
            )
            count += 1

    @cached_property
    def topic_recordtype(self):
        return record_type(ir.IDItem)

    @cached_property
    def document_recordtype(self):
        return record_type(ir.IDItem)


class FakeTextStore(TextStore):
    def __getitem__(self, key: str) -> str:
        return f"T{key}"


class FakeDocumentStore(ir.DocumentStore):
    id: Param[str] = ""

    def document_ext(self, docid: str) -> ir.DocumentRecord:
        return ir.create_record(id=docid, text=f"D{docid}")


class _PairwiseDataset(ShardedIterableDataset):
    """Simple dataset yielding ID-only PairwiseRecords for testing."""

    def iter_shard(self, shard_id, num_shards):
        count = 0
        while True:
            yield PairwiseRecord(
                ir.create_record(id=str(count)),
                ir.create_record(id=str(2 * count)),
                ir.create_record(id=str(2 * count + 1)),
            )
            count += 1


def test_store_hydrator_process_batch():
    """Test that StoreHydrator correctly hydrates a batch of PairwiseRecords."""
    hydrator = StoreHydrator.C(
        documentstore=FakeDocumentStore.C(),
        querystore=FakeTextStore.C(),
    ).instance()

    records = [
        PairwiseRecord(
            ir.create_record(id=str(n)),
            ir.create_record(id=str(2 * n)),
            ir.create_record(id=str(2 * n + 1)),
        )
        for n in range(5)
    ]

    hydrated = hydrator.process_batch(records)
    assert len(hydrated) == 5
    for n, record in enumerate(hydrated):
        assert record.query[ir.TextItem].text == f"T{n}"
        assert record.positive[ir.TextItem].text == f"D{2 * n}"
        assert record.negative[ir.TextItem].text == f"D{2 * n + 1}"


def test_buffered_processing_dataset():
    """Test that BufferedProcessingDataset buffers and hydrates correctly."""
    hydrator = StoreHydrator.C(
        documentstore=FakeDocumentStore.C(),
        querystore=FakeTextStore.C(),
    ).instance()

    inner_dataset = _PairwiseDataset()
    dataset = BufferedProcessingDataset(inner_dataset, [hydrator], buffer_size=4)

    count = 0
    for record in dataset:
        n = count
        assert record.query[ir.TextItem].text == f"T{n}"
        assert record.positive[ir.TextItem].text == f"D{2 * n}"
        assert record.negative[ir.TextItem].text == f"D{2 * n + 1}"
        count += 1
        if count >= 10:
            break

    assert count == 10
