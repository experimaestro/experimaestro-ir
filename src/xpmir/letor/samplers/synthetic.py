from pathlib import Path
import json
from typing import Any, Callable, Optional, Tuple, Iterator, List
import numpy as np

from experimaestro import Task, Param, Meta, Annotated, pathgenerator, tqdm
from datamaestro_text.data.ir import DocumentStore
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

    documents: Param[DocumentStore]
    """The set of documents"""

    batchsize: Param[int] = 128

    num_qry_per_doc: Param[int] = 5
    """How many synthetic qry to generate per document"""

    sampler: Param[Optional[DocumentSampler]]
    """Optional document sampler when training the index -- by default, all the
    documents from the collection are used"""

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
        # TODO: read the file and pass it to the sampler
        pass

    def full_sampler(self) -> Tuple[int, Iterator[str]]:
        """Returns an iterator over the full set of documents"""
        internal_ids = np.arange(self.documents.documentcount)
        np.random.shuffle(internal_ids)
        id_list = internal_ids.tolist()

        iter = (
            (self.documents.document_int(id).id, self.documents.document_int(id).text)
            for id in id_list
        )
        return self.documents.documentcount or 0, iter

    def execute(self):
        self.device.execute(self.device_execute)

    def generate(self, batch: List[Tuple[str, str]], fp):
        generate_output = self.model.generate(
            [d[1] for d in batch], self.generation_config
        )
        # length: bs*num_qry_per_doc
        queries = self.model.batch_decode(generate_output)

        # group the queries corresponds to the same document together.
        grouped_queries = [
            queries[i : i + self.num_qry_per_doc]
            for i in range(0, len(queries), self.num_qry_per_doc)
        ]
        doc_ids = [d[0] for d in batch]

        for qry, doc_id in zip(grouped_queries, doc_ids):
            dict_query_doc = dict()
            dict_query_doc["queries"] = qry
            dict_query_doc["pos_ids"] = [doc_id]
            dict_query_doc["neg_ids"] = []
            json.dump(dict_query_doc, fp)
            fp.write("\n")

    def device_execute(self, device_information: DeviceInformation):
        count, iter = (
            self.sampler() if self.sampler is not None else self.full_sampler()
        )
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
