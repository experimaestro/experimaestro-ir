from pathlib import Path
import json
from typing import Any, Callable, List, Optional, Dict

import numpy as np
from experimaestro import Task, Param, Meta, Annotated, pathgenerator, tqdm
from experimaestro.core.objects import Config
from datamaestro_text.data.ir import DocumentStore

from xpmir.context import Context, Hook, InitializationHook
from xpmir.documents.samplers import DocumentSampler
from xpmir.neural.generative import BeamSearchGenerationOptions
from xpmir.neural.generative.hf import T5ConditionalGenerator
from xpmir.letor import Random
from xpmir.letor.samplers import JSONLPairwiseSampleDataset
from xpmir.rankers import Retriever

from xpmir.learning import ModuleInitMode
from xpmir.learning.batchers import Batcher
from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from xpmir.utils.utils import batchiter, easylog, foreach

logger = easylog()


class SyntheticQueryGeneration(Task):

    model: Param[T5ConditionalGenerator]
    """The model we use to generate the queries"""

    batchsize: Meta[int] = 128
    """Batchsize when computing negatives"""

    num_qry_per_doc: Param[int] = 5
    """How many synthetic qry to generate per document"""

    sampler: Param[DocumentSampler]
    """document sampler to iterate over the corpus"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device used by the encoder"""

    batcher: Meta[Batcher] = Batcher()
    """The way to prepare batches of documents"""

    synthetic_samples: Annotated[Path, pathgenerator("synthetic.jsonl")]
    """Path to store the generated queries"""

    hooks: Param[List[Hook]] = []
    """Global learning hooks"""

    def __post_init__(self):
        super().__post_init__()
        self.generation_config = BeamSearchGenerationOptions(
            num_return_sequences=self.num_qry_per_doc,
            num_beams=self.num_qry_per_doc,
            max_new_tokens=64,
        )

    def task_outputs(self, dep: Callable[[Config], None]) -> Any:
        return dep(
            JSONLPairwiseSampleDataset(
                id=self.sampler.documents.id,
                path=self.synthetic_samples,
            )
        )

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
            dict_query_doc["neg_ids"] = {"random": []}
            json.dump(dict_query_doc, fp)
            fp.write("\n")

    def device_execute(self, device_information: DeviceInformation):
        # Initialization hooks
        context = Context(device_information, hooks=self.hooks)
        foreach(context.hooks(InitializationHook), lambda hook: hook.before(context))

        count, iter = self.sampler()
        doc_iter = tqdm(iter, total=count, desc="Generating the synthetic queries")

        self.model.initialize(ModuleInitMode.DEFAULT.to_options())
        # put the model on the device and eval mode
        self.model.to(device_information.device).eval()
        batcher = self.batcher.initialize(self.batchsize)

        # Initialization hooks (after)
        foreach(context.hooks(InitializationHook), lambda hook: hook.after(context))

        # generate the synthetic tokens
        with self.synthetic_samples.open("wt") as fp:
            for batch in batchiter(self.batchsize, doc_iter):
                batcher.process(batch, self.generate, fp)


class JSONLNegativeGeneration(Task):
    """Add the negatives to the pairwise sampler according to the given retrievers."""

    random: Param[Optional[Random]] = None
    """The random sampler"""

    documents: Param[DocumentStore]
    """The document store where the negatives are sampling from"""

    pairwise_dataset: Param[JSONLPairwiseSampleDataset]
    """The pairwise dataset where we are going to add the negatives"""

    retrievers: Param[Dict[str, Retriever]]
    """The retrievers to retrieve the top k document wrt the query, if no
    retriever's provided, we just use the random negatives"""

    synthetic_samples: Annotated[Path, pathgenerator("synthetic_negatives.jsonl")]
    """Path to store the generated queries"""

    k: Param[int] = 100
    """The number of negatives for each algo"""

    def task_outputs(self, dep: Callable[[Config], None]) -> Any:
        return dep(
            JSONLPairwiseSampleDataset(
                id=self.documents.id,
                path=self.synthetic_samples,
            )
        )

    def execute(self):
        for retriever in self.retrievers.values():
            retriever.initialize()

        logger.info("Start to generate the negatives")

        pairwise_sample_iter = tqdm(
            self.pairwise_dataset.iter(),
            total=self.pairwise_dataset.count,
            desc="Generating negatives for the JSONL",
        )

        with self.synthetic_samples.open("wt") as fp:
            for pairwise_sample in pairwise_sample_iter:
                dict_query_doc = dict()
                query_texts = [q.text for q in pairwise_sample.topics]
                positive_ids = [pos.id for pos in pairwise_sample.positives]
                dict_query_doc["queries"] = query_texts
                dict_query_doc["pos_ids"] = positive_ids

                negatives = {}
                state = (
                    np.random.RandomState()
                    if self.random is None
                    else self.random.state
                )

                # Retrieve based on the algo
                # TODO: Make it in batch
                for (algo_name, retriever) in self.retrievers.items():
                    query_text = query_texts[state.randint(len(query_texts))]
                    scoreddocuments = retriever.retrieve(query_text)
                    ext_ids = [sd.document.id for sd in scoreddocuments]
                    filitered = [
                        ext_id for ext_id in ext_ids if ext_id not in positive_ids
                    ]
                    negatives[algo_name] = filitered
                dict_query_doc["neg_ids"] = negatives
                json.dump(dict_query_doc, fp)
                fp.write("\n")
