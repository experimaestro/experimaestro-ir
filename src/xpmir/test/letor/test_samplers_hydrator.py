import itertools
from experimaestro import Param
from typing import Iterator, Tuple
import datamaestro_text.data.ir as ir
from datamaestro_text.data.ir.base import (
    Document,
    IDTopic,
    IDDocument,
    GenericDocument,
)
from xpmir.letor.samplers import (
    TrainingTriplets,
    TripletBasedSampler,
)
from xpmir.datasets.adapters import TextStore

from xpmir.letor.samplers.hydrators import (
    SampleHydrator,
    SampleTransformList,
    PairwiseTransformAdapter,
)


class TripletIterator(TrainingTriplets):
    def iter(self) -> Iterator[Tuple[IDTopic, IDDocument, IDDocument]]:
        count = 0

        while True:
            yield IDTopic(str(count)), IDDocument(str(2 * count)), IDDocument(
                str(2 * count + 1)
            )
            count += 1


class FakeTextStore(TextStore):
    def __getitem__(self, key: str) -> str:
        return f"T{key}"


class FakeDocumentStore(ir.DocumentStore):
    id: Param[str] = ""

    def document_ext(self, docid: str) -> Document:
        return GenericDocument(docid, f"D{docid}")


def test_pairwise_hydrator():
    sampler = TripletBasedSampler(source=TripletIterator(id="test-triplets"))

    adapters = SampleTransformList(
        adapters=[
            SampleHydrator(
                querystore=FakeTextStore(), documentstore=FakeDocumentStore()
            )
        ]
    )

    h_sampler = PairwiseTransformAdapter(sampler=sampler, adapter=adapters)
    h_sampler.instance()

    for record, n in zip(h_sampler.pairwise_iter(), range(5)):
        assert record.query.topic.get_text() == f"T{n}"
        assert record.positive.document.get_text() == f"D{2*n}"
        assert record.negative.document.get_text() == f"D{2*n+1}"

    batch_it = h_sampler.pairwise_batch_iter(3)
    for record, n in zip(itertools.chain(next(batch_it), next(batch_it)), range(5)):
        assert record.query.topic.get_text() == f"T{n}"
        assert record.positive.document.get_text() == f"D{2*n}"
        assert record.negative.document.get_text() == f"D{2*n+1}"
