from typing import Any, Callable, Optional, Tuple, Iterator
from experimaestro import Task, Param, Meta
from datamaestro_text.data.ir import DocumentStore
from experimaestro.core.objects import Config

from xpmir.documents.samplers import DocumentSampler
from xpmir.letor.samplers import PairwiseSampler
from xpmir.neural.generative.hf import T5ConditionalGenerator
from xpmir.learning import ModuleInitMode
from xpmir.learning.batchers import Batcher
from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation


class SynetheticQueryGeneration(Task):

    model: Param[T5ConditionalGenerator]
    """The model we use to generate the queries"""

    documents: Param[DocumentStore]
    """The set of documents"""

    batchsize: Param[int] = 128

    sampler: Param[Optional[DocumentSampler]]
    """Optional document sampler when training the index -- by default, all the
    documents from the collection are used"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device used by the encoder"""

    batcher: Meta[Batcher] = Batcher()
    """The way to prepare batches of documents"""

    def task_outputs(self, dep: Callable[[Config], None]) -> Any:
        # TODO: read the file and pass it to the sampler
        return super().task_outputs(dep)

    def full_sampler(self) -> Tuple[int, Iterator[str]]:
        """Returns an iterator over the full set of documents"""
        iter = (d.text for d in self.documents.iter_documents())
        return self.documents.documentcount or 0, iter

    def execute(self):
        self.device.execute(self.device_execute)

    def device_execute(self, device_information: DeviceInformation):
        self.model.initialize(ModuleInitMode.DEFAULT.to_options())
        # put the model on the device and eval mode
        self.model.to(device_information.device).eval()
        batcher = self.batcher.initialize(self.batchsize)

        # TODO: generate the synthetic tokens
        pass
