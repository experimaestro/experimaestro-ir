from typing import Any, Callable, Annotated, List
from pathlib import Path
from experimaestro import Task, Param, Config, pathgenerator, Meta, tqdm
import torch
import numpy as np
import json
import faiss


from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from xpmir.learning.batchers import Batcher
from xpmir.learning import ModuleInitMode
from xpmir.text.encoders import TokenizedEncoder
from xpmir.text.tokenizers import TokenizerBase
from xpmir.documents.samplers import DocumentSampler

from xpmir.utils.utils import batchiter, easylog

logger = easylog()


class ReferentialFixDocumentIdBuilder(Task):

    sampler: Param[DocumentSampler]
    """document sampler to iterate over the corpus"""

    encoder: Param[TokenizedEncoder]
    """The encoder to encode the documents"""

    tokenizer: Param[TokenizerBase]
    """The tokenizer to tokenize the text"""

    decoder_dim: Param[int]
    """The decoder_dim of the referential model"""

    max_depth: Param[int]
    """The max depth of the referential model"""

    batchsize: Param[int] = 256
    """The batchsize for the encoder"""

    index_path: Annotated[Path, pathgenerator("index.pth")]
    """Path to the file containing the index"""

    target_path: Annotated[Path, pathgenerator("target.jsonl")]
    """Path to the file contains the target"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device used by the encoder"""

    batcher: Meta[Batcher] = Batcher()
    """The way to prepare batches of documents"""

    alpha: Param[int] = 39
    """The threshold for stopping the hierarchical clustering"""

    def task_outputs(self, dep: Callable[[Config], None]) -> Any:
        pass

    def index_documents(self, batch: List[str], data):
        tokenized = self.tokenizer.tokenize(batch)
        x = self.encoder(tokenized).value
        # FIXME: better not do it here
        mask = tokenized.mask.to(self.device.value)
        input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        res = torch.sum(x * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
        data.append(res.cpu().detach())

    def execute(self) -> None:
        self.device.execute(self.device_execute)

    def device_execute(self, device_information: DeviceInformation):
        if not self.index_path.is_file():
            self.encoder.initialize(ModuleInitMode.DEFAULT.to_options())
            self.tokenizer.initialize(ModuleInitMode.DEFAULT.to_options())
            batcher = self.batcher.initialize(self.batchsize)
            # Change the device of the encoder
            self.encoder.to(device_information.device).eval()

            count, iter = self.sampler()
            doc_iter = tqdm(iter, total=count, desc="Building the index")

            with torch.no_grad():
                index = []
                for batch in batchiter(self.batchsize, doc_iter):
                    batcher.process(
                        [document.get_text() for document in batch],
                        self.index_documents,
                        index,
                    )
                self.index = torch.cat(index)

            logger.info("Writing FAISS index (%d documents)", self.index.shape[0])
            torch.save(self.index, self.index_path)
        else:
            logger.info(
                "Index already exist, start the hierarchical k-means clustering"
            )
            self.index = torch.load(self.index_path)

        # start hierarchical k-means clustering
        self.index = self.index.cpu().numpy()
        input_indice = np.arange(self.index.shape[0])

        with self.target_path.open("wt") as fp:
            self.classify_recursion(1, input_indice, self.max_depth, [], fp)

    def classify_recursion(
        self,
        depth,  # current depth
        data_indice,  # the indices of the original data where we can start the k-means
        max_depth,  # where to stop?
        prefix,  # a list, which signify the prefix
        fp,  # the file
    ):
        data_to_kmeans = self.index[data_indice]

        # test end condition
        # if the number of the number of tokens is smaller than k, no need to
        # k-means clustering
        if depth == max_depth + 1 or data_indice.shape[0] < int(
            self.alpha * self.decoder_dim
        ):
            if data_indice.shape[0] == 0:
                return
            output = {}
            prefix_str = "\t".join(map(str, prefix))
            output[prefix_str] = [
                self.sampler.documents.docid_internal2external(i)
                for i in data_indice.tolist()
            ]
            json.dump(output, fp)
            fp.write("\n")
            return

        if len(prefix) == 0:
            logger.info("Start the hierarchical k-means clustering from the root")
        else:
            logger.info(f"Start the hierarchical k-means clustering for {prefix}")

        kmeans = faiss.Kmeans(
            self.index.shape[1], self.decoder_dim, niter=300, gpu=True
        )
        kmeans.train(data_to_kmeans)
        _, INDICE = kmeans.index.search(data_to_kmeans, 1)

        for i in range(self.decoder_dim):
            new_prefix = prefix.copy()
            new_prefix.append(i)
            # prepare the new indices.
            new_data_indice = data_indice[np.where(INDICE.reshape(-1) == i)[0]]
            self.classify_recursion(
                depth + 1, new_data_indice, max_depth, new_prefix, fp
            )
