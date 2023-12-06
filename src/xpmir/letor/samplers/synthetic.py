from typing import Any, Callable
from xpmir.rankers import Retriever
from experimaestro import Config, Task, Param, Meta
from datamaestro_text.data.ir import DocumentStore
from xpmir.learning.devices import DEFAULT_DEVICE, Device


class SyntheticTripletGenerator(Task):
    """Using the synthetic triplets for data augmentation
    Cf. InPars http://arxiv.org/abs/2202.05144
    General Algo: Use a model to generate queries based on a given document
    then use a retriever to get the 'hard' negatives by using the generated
    query
    """

    device: Meta[Device] = DEFAULT_DEVICE
    """The device(s) to be used for the model"""

    d2q_id: Param[str] = "doc2query/msmarco-t5-base-v1"
    """The hf id to generate the query"""

    retriever: Param[Retriever]
    """The retriever to retrieve the negatives"""

    dataset: Param[DocumentStore]
    """The document store to do the data augmentation"""

    def task_outputs(self, dep: Callable[[Config], None]) -> Any:
        pass

    def execute(self):
        self.device.execute(self.device_execute)

    def device_execute():
        pass
