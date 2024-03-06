from functools import cached_property
import itertools
from experimaestro import Param
from typing import Iterator, Tuple
from datamaestro.record import record_type
import datamaestro_text.data.ir as ir
from xpmir.letor.samplers import (
    TrainingTriplets,
    TripletBasedSampler,
)
from xpmir.datasets.adapters import TextStore

from xpmir.letor.samplers.hydrators import (
    SampleHydrator,
    PairwiseTransformAdapter,
)


class TripletIterator(TrainingTriplets):
    def iter(
        self,
    ) -> Iterator[Tuple[ir.TopicRecord, ir.DocumentRecord, ir.DocumentRecord]]:
        count = 0

        while True:
            yield ir.create_record(id=str(count)), ir.create_record(
                id=str(2 * count)
            ), ir.create_record(id=str(2 * count + 1))
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


def test_pairwise_hydrator():
    sampler = TripletBasedSampler(source=TripletIterator(id="test-triplets"))

    hydrator = SampleHydrator(
        querystore=FakeTextStore(), documentstore=FakeDocumentStore()
    )

    h_sampler = PairwiseTransformAdapter(sampler=sampler, adapter=hydrator)
    h_sampler.instance()

    for record, n in zip(h_sampler.pairwise_iter(), range(5)):
        assert record.query[ir.TextItem].text == f"T{n}"
        assert record.positive[ir.TextItem].text == f"D{2*n}"
        assert record.negative[ir.TextItem].text == f"D{2*n+1}"

    batch_it = h_sampler.pairwise_batch_iter(3)
    for record, n in zip(itertools.chain(next(batch_it), next(batch_it)), range(5)):
        assert record.query[ir.TextItem].text == f"T{n}"
        assert record.positive[ir.TextItem].text == f"D{2*n}"
        assert record.negative[ir.TextItem].text == f"D{2*n+1}"
