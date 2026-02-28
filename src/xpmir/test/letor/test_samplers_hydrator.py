from functools import cached_property
import itertools
from experimaestro import Param
from typing import Iterator, Tuple
from datamaestro_text.data.ir import IDTextRecord, SimpleTextItem
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
    ) -> Iterator[Tuple[IDTextRecord, IDTextRecord, IDTextRecord]]:
        count = 0

        while True:
            yield (
                {"id": str(count)},
                {"id": str(2 * count)},
                {"id": str(2 * count + 1)},
            )
            count += 1


class FakeTextStore(TextStore):
    def __getitem__(self, key: str) -> str:
        return f"T{key}"


class FakeDocumentStore(ir.DocumentStore):
    id: Param[str] = ""

    def document_ext(self, docid: str) -> IDTextRecord:
        return {"id": docid, "text_item": SimpleTextItem(f"D{docid}")}


def test_pairwise_hydrator():
    sampler = TripletBasedSampler.C(source=TripletIterator.C(id="test-triplets"))

    hydrator = SampleHydrator.C(
        querystore=FakeTextStore.C(), documentstore=FakeDocumentStore.C()
    )

    h_sampler = PairwiseTransformAdapter.C(sampler=sampler, adapter=hydrator)
    h_sampler.instance()

    for record, n in zip(h_sampler.pairwise_iter(), range(5)):
        assert record.query["text_item"].text == f"T{n}"
        assert record.positive["text_item"].text == f"D{2*n}"
        assert record.negative["text_item"].text == f"D{2*n+1}"

    batch_it = h_sampler.pairwise_batch_iter(3)
    for record, n in zip(itertools.chain(next(batch_it), next(batch_it)), range(5)):
        assert record.query["text_item"].text == f"T{n}"
        assert record.positive["text_item"].text == f"D{2*n}"
        assert record.negative["text_item"].text == f"D{2*n+1}"
