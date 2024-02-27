import itertools
from experimaestro import Param
from typing import Iterator, Tuple
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
    ) -> Iterator[Tuple[ir.IDTopicRecord, ir.IDDocumentRecord, ir.IDDocumentRecord]]:
        count = 0

        while True:
            yield ir.IDTopicRecord.from_id(str(count)), ir.IDDocumentRecord.from_id(
                str(2 * count)
            ), ir.IDDocumentRecord.from_id(str(2 * count + 1))
            count += 1

    @property
    def topic_recordtype(self):
        return ir.IDTopicRecord

    @property
    def document_recordtype(self):
        return ir.IDDocumentRecord


class FakeTextStore(TextStore):
    def __getitem__(self, key: str) -> str:
        return f"T{key}"


class FakeDocumentStore(ir.DocumentStore):
    id: Param[str] = ""

    def document_ext(self, docid: str) -> ir.GenericDocumentRecord:
        return ir.GenericDocumentRecord.create(docid, f"D{docid}")


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
