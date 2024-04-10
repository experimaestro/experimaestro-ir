from typing import Any, Callable, Annotated, List
from pathlib import Path
from experimaestro import Task, Param, Config, pathgenerator, Meta, tqdm
import torch
import numpy as np
import json
import faiss

from datamaestro_text.data.ir import TextItem, IDItem

from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from xpmir.learning.batchers import Batcher
from xpmir.learning import ModuleInitMode
from xpmir.text.encoders import TokenizedEncoder
from xpmir.text.tokenizers import TokenizerBase
from xpmir.documents.samplers import DocumentSampler
from xpmir.neural.generative.referential.samplers import (
    JSONLReferentialDocumentIdDataset,
)

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

    beta: Param[int] = 8
    """The threshold for the minimal number of documents for each sequence to
    avoid overfitting."""

    def task_outputs(self, dep: Callable[[Config], None]) -> Any:
        return dep(
            JSONLReferentialDocumentIdDataset(
                path=self.target_path,
            )
        )

    def index_documents(self, batch: List[str], data):
        tokenized = self.tokenizer.tokenize(batch)
        x = self.encoder(tokenized).value
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
                        [document[TextItem].text for document in batch],
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
    ) -> List[int]:
        data_to_kmeans = self.index[data_indice]

        # test end condition
        # if the number of the number of tokens is smaller than a threshold, no need to
        # k-means clustering
        if depth == max_depth + 1 or data_indice.shape[0] < int(
            self.alpha * self.decoder_dim
        ):
            if data_indice.shape[0] == 0:
                return []
            output = {}
            prefix_str = "\t".join(map(str, prefix))
            output[prefix_str] = [
                self.sampler.documents.docid_internal2external(i)
                for i in data_indice.tolist()
            ]
            json.dump(output, fp)
            fp.write("\n")
            return []

        if len(prefix) == 0:
            logger.info("Start the hierarchical k-means clustering from the root")
        else:
            logger.info(f"Start the hierarchical k-means clustering for {prefix}")

        kmeans = faiss.Kmeans(
            self.index.shape[1], self.decoder_dim, niter=300, gpu=False
        )
        kmeans.train(data_to_kmeans)
        _, INDICE = kmeans.index.search(data_to_kmeans, 1)

        # the list of the documents for the current_cluster
        current_cluster = []
        for i in range(self.decoder_dim):
            new_prefix = prefix.copy()
            new_prefix.append(i)
            # prepare the new indices.
            new_data_indice = data_indice[np.where(INDICE.reshape(-1) == i)[0]]
            # e.g. if the document belongs to 2.4.5.1 is smaller than a
            # threshold, we consider them belongs to 2.4.5 to avoid overfitting,
            # if still too small, consider it belongs to 2.4, etc.
            if new_data_indice.shape[0] < self.beta:
                current_cluster.extend(new_data_indice.tolist())
                logger.info(
                    f"The cluster {new_prefix} doesn't contains enough \
                    documents(number: {new_data_indice.shape[0]}), trying \
                    to store them inside the {prefix}"
                )
            else:
                deeper_layer_not_classified = self.classify_recursion(
                    depth + 1, new_data_indice, max_depth, new_prefix, fp
                )
                current_cluster.extend(deeper_layer_not_classified)

        # check if the documents for the current cluster achieve the threshold
        # if true store them, if not return to the previous layers
        # Or if the depth == 1, we need to store them also because otherwise
        # we will have no place to store them anymore
        if len(current_cluster) >= self.beta or depth == 1:
            output = {}
            prefix_str = "\t".join(map(str, prefix))
            output[prefix_str] = [
                self.sampler.documents.docid_internal2external(i)
                for i in current_cluster
            ]
            json.dump(output, fp)
            fp.write("\n")
            return []
        else:
            if len(current_cluster) != 0:
                logger.info(
                    f"No enough documents for {prefix}(number: \
                    {len(current_cluster)}), trying to store them \
                    in their father"
                )
            return current_cluster


class RandomSplitReferentialFixDocumentID(Task):

    dataset: Param[JSONLReferentialDocumentIdDataset]
    """The original dataset"""

    split_size: Param[int] = 1000
    """The size for the splited dataset"""

    big: Annotated[Path, pathgenerator("big.jsonl")]
    """Path to the file contains the majarity of the """

    small: Annotated[Path, pathgenerator("small.jsonl")]
    """Path to the file contains the target"""

    def task_outputs(self, dep: Callable[[Config], None]) -> Any:
        return (
            dep(JSONLReferentialDocumentIdDataset(path=self.big)),
            dep(JSONLReferentialDocumentIdDataset(path=self.small)),
        )

    def execute(self) -> None:
        # rows to select the for the small dataset
        rows = np.random.choice(
            self.dataset.count,
            size=min(self.split_size, self.dataset.count),
            replace=False,
        )
        current_count = 0
        with self.small.open("wt") as fp1, self.big.open("wt") as fp2:
            for sample in tqdm(self.dataset.iter(), total=self.dataset.count):
                dict_big = {}
                id_str = "\t".join(map(str, sample.ids))
                doc_id_list = [doc[IDItem].id for doc in sample.documents]
                if current_count in rows:
                    dict_small = {}
                    sampled = doc_id_list[np.random.randint(len(doc_id_list))]
                    doc_id_list = [x for x in doc_id_list if x != sampled]
                    dict_small[id_str] = [sampled]
                    json.dump(dict_small, fp1)
                    fp1.write("\n")

                dict_big[id_str] = doc_id_list
                json.dump(dict_big, fp2)
                fp2.write("\n")
                current_count += 1
