from pathlib import Path
import json
from typing import Any, Callable

from experimaestro import Task, Param, Meta, Annotated, pathgenerator, tqdm
from experimaestro.core.objects import Config

from xpmir.documents.samplers import DocumentSampler
from xpmir.neural.generative import BeamSearchGenerationOptions
from xpmir.neural.generative.hf import T5ConditionalGenerator
from xpmir.learning import ModuleInitMode
from xpmir.learning.batchers import Batcher
from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from xpmir.utils.utils import batchiter, easylog

logger = easylog()


class SynetheticQueryGeneration(Task):

    model: Param[T5ConditionalGenerator]
    """The model we use to generate the queries"""

    batchsize: Param[int] = 128

    num_qry_per_doc: Param[int] = 5
    """How many synthetic qry to generate per document"""

    sampler: Param[DocumentSampler]
    """document sampler to iterate over the corpus"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device used by the encoder"""

    batcher: Meta[Batcher] = Batcher()
    """The way to prepare batches of documents"""

    synthetic_samples: Annotated[Path, pathgenerator("synthetic.jsonl")]
    """Path to store the generated hard negatives"""

    def __post_init__(self):
        super().__post_init__()
        self.generation_config = BeamSearchGenerationOptions(
            num_return_sequences=self.num_qry_per_doc,
            num_beams=self.num_qry_per_doc,
            max_new_tokens=64,
        )

    def task_outputs(self, dep: Callable[[Config], None]) -> Any:
        return self.synthetic_samples

    def execute(self):
        self.device.execute(self.device_execute)

    def generate(self, batch, fp):
        generate_output = self.model.generate(
            [d.text for d in batch], self.generation_config
        )
        # length: bs*num_qry_per_doc
        queries = self.model.batch_decode(generate_output)

        # group the queries corresponds to the same document together.
        grouped_queries = [
            queries[i : i + self.num_qry_per_doc]
            for i in range(0, len(queries), self.num_qry_per_doc)
        ]
        doc_ids = [d.id for d in batch]

        for qry, doc_id in zip(grouped_queries, doc_ids):
            dict_query_doc = dict()
            dict_query_doc["queries"] = qry
            dict_query_doc["pos_ids"] = [doc_id]
            dict_query_doc["neg_ids"] = []
            json.dump(dict_query_doc, fp)
            fp.write("\n")

    def device_execute(self, device_information: DeviceInformation):
        count, iter = self.sampler()
        doc_iter = tqdm(
            iter, total=count, desc="Collecting the representation of documents (train)"
        )

        self.model.initialize(ModuleInitMode.DEFAULT.to_options())
        # put the model on the device and eval mode
        self.model.to(device_information.device).eval()
        batcher = self.batcher.initialize(self.batchsize)

        # generate the synthetic tokens
        with self.synthetic_samples.open("wt") as fp:
            for batch in batchiter(self.batchsize, doc_iter):
                batcher.process(batch, self.generate, fp)
