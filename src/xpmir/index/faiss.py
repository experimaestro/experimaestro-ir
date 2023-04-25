"""Interface to the Facebook FAISS library

https://github.com/facebookresearch/faiss
"""
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Tuple, Dict
from experimaestro import Config, initializer
import torch
import numpy as np
from experimaestro import Annotated, Meta, Task, pathgenerator, Param, tqdm
import logging
from datamaestro_text.data.ir import AdhocDocumentStore
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.learning.batchers import Batcher
from xpmir.text.encoders import TextEncoder
from xpmir.letor import (
    Device,
    DEFAULT_DEVICE,
    DeviceInformation,
)
from xpmir.utils.utils import batchiter, easylog, foreach
from xpmir.documents.samplers import DocumentSampler
from xpmir.context import Context, Hook, InitializationHook
from xpmir.letor.learner import LearnerListener, Learner
from xpmir.letor.context import TrainerContext, TrainState

logger = easylog()

try:
    import faiss
except ModuleNotFoundError:
    logging.error("FAISS library is not available (install faiss-cpu or faiss)")
    raise


class BaseFaissIndex(Config):
    normalize: Param[bool]
    """Whether vectors should be normalized (L2)"""

    documents: Param[AdhocDocumentStore]
    """The set of documents"""

    def get_index(self):
        raise NotImplementedError(f"get_index in {self.__class__}")


class FaissIndex(BaseFaissIndex):
    """FAISS Index, static version, where the index are fixed once finish the
    computation"""

    faiss_index: Param[Path]
    """Path to the file containing the index"""

    @lru_cache
    def get_index(self):
        return faiss.read_index(str(self.faiss_index))


class BaseIndexBackedFaiss(Config):
    """Constructs a FAISS index backed up by an index

    During executions, InitializationHooks are used (pre/post)

    """

    normalize: Param[bool]
    """Whether vectors should be normalized (L2)"""

    documents: Param[AdhocDocumentStore]
    """The set of documents"""

    encoder: Param[TextEncoder]
    """Encoder for document texts"""

    batchsize: Meta[int] = 1
    """The batch size used when computing representations of documents"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device used by the encoder"""

    batcher: Meta[Batcher] = Batcher()
    """The way to prepare batches of documents"""

    hooks: Param[List[Hook]] = []
    """An optional list of hooks"""

    indexspec: Param[str]
    """The index type as a factory string

    See https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
    for the full list of indices

    and https://github.com/facebookresearch/faiss/wiki/The-index-factory
    for the combination of the index factory
    """

    sampler: Param[Optional[DocumentSampler]]
    """Optional document sampler when training the index -- by default, all the
    documents from the collection are used"""

    def full_sampler(self) -> Tuple[int, Iterator[str]]:
        """Returns an iterator over the full set of documents"""
        iter = (d.text for d in self.documents.iter_documents())
        return self.documents.documentcount or 0, iter

    def train(
        self,
        index: faiss.Index,
        batch_encoder: Callable[[Iterator[str]], Iterator[torch.Tensor]],
    ):
        """train the index

        params
        index:
        batch_encoder:
            function, input is a iterator of list of documents str, return the
            encoded document vector(tensor of shape (bs*dimension))
        """
        logger.info("Building index")
        count, iter = (
            self.sampler() if self.sampler is not None else self.full_sampler()
        )
        doc_iter = tqdm(
            iter, total=count, desc="Collecting the representation of documents (train)"
        )

        # Collect batches (in memory)
        logger.info("Collecting the representation of %d documents", count)
        sample = np.ndarray((count, self.encoder.dimension), dtype=np.float32)
        ix = 0
        for batch in batch_encoder(doc_iter):
            sample[ix : (ix + len(batch))] = batch.cpu().numpy()
            ix += len(batch)

        logger.info("Training index (%d samples)", count)
        # Here we may use just a part of the document to train the index
        index.train(sample)

    def _execute(self, faiss_index: Path, device_information: DeviceInformation):
        # Initialization hooks
        context = Context(device_information, hooks=self.hooks)
        foreach(context.hooks(InitializationHook), lambda hook: hook.before(context))

        step_iter = tqdm(total=2, desc="Building the FAISS index")

        # Initializations
        self.encoder.initialize()
        index = faiss.index_factory(
            self.encoder.dimension, self.indexspec, faiss.METRIC_INNER_PRODUCT
        )
        batcher = self.batcher.initialize(self.batchsize)

        # Change the device of the encoder
        self.encoder.to(device_information.device).eval()

        # Train the index
        if not index.is_trained:
            with torch.no_grad():
                logging.info("Training FAISS index (%d documents)", index.ntotal)

                def batch_encoder(doc_iter: Iterator[str]):
                    for batch in batchiter(self.batchsize, doc_iter):
                        data = []
                        batcher.process(batch, self.encode, data)
                        yield torch.cat(data)

                self.train(index, batch_encoder)

        step_iter.update()

        # Index the collection
        doc_iter = (
            tqdm(
                self.documents.iter_documents(),
                total=self.documents.documentcount,
                desc="Indexing the collection",
            )
            if device_information.main
            else self.documents.iter_documents()
        )

        # Initialization hooks (after)
        foreach(context.hooks(InitializationHook), lambda hook: hook.after(context))

        # Let's index !
        # We add index for all the documents
        with torch.no_grad():
            for batch in batchiter(self.batchsize, doc_iter):
                batcher.process(
                    [document.text for document in batch], self.index_documents, index
                )

        logging.info("Writing FAISS index (%d documents)", index.ntotal)
        faiss.write_index(index, str(faiss_index))
        step_iter.update()

    def encode(self, batch: List[str], data: List):
        batch = [
            text for text in batch if text != ""
        ]  # remove the empty strings in the dataset (training only)
        x = self.encoder(batch)
        if self.normalize:
            x /= x.norm(2, keepdim=True, dim=1)
        data.append(x)

    def index_documents(self, batch: List[str], index):
        x = self.encoder(batch)
        if self.normalize:
            x /= x.norm(2, keepdim=True, dim=1)
        index.add(np.ascontiguousarray(x.cpu().numpy()))

    def docid_internal2external(self, docid: int):
        return self.documents.docid_internal2external(docid)


class DynamicFaissIndex(BaseIndexBackedFaiss, BaseFaissIndex):
    """A faiss index which could be updated through the listener"""

    def get_index(self):
        if self._index is None:
            self._index = faiss.read_index(str(self.faiss_index))
        return self._index

    def update(self, path: Path):
        # Update the index and store it at the path
        # execute the code to generate the faiss index.
        self.device.execute(partial(self._execute, path))
        self.faiss_index = path

        # clear the previous cache
        self._index = None


class IndexBackedFaiss(BaseIndexBackedFaiss, Task):
    """Constructs a FAISS index backed up by an index

    During executions, InitializationHooks are used (pre/post)

    """

    faiss_index: Annotated[Path, pathgenerator("faiss.dat")]
    """Path to the file containing the index"""

    def execute(self):
        self.device.execute(partial(self._execute, self.faiss_index))

    def taskoutput(self):
        return FaissIndex(
            normalize=self.normalize,
            documents=self.documents,
            faiss_index=self.faiss_index,
        )


class FaissBuildListener(LearnerListener):

    indexing_interval: Param[int] = 128
    """During how many epochs we recompute the index"""

    indexbackedfaiss: Param[DynamicFaissIndex]
    """The faiss object"""

    def initialize(self, learner: "Learner", context: TrainerContext):
        super().initialize(learner, context)

    def __call__(self, state: TrainState) -> bool:

        if state.epoch % self.indexing_interval == 0:

            # state.path = 'checkpoint/epoch-00000XX/'
            path = state.path / "listeners" / self.id / "faiss.dat"
            path.mkdir(exist_ok=True, parents=True)
            self.indexbackedfaiss.update(path)

        return False

    def update_metrics(self, metrics: Dict[str, float]):
        pass

    def taskoutputs(self, learner: "Learner"):
        pass


class FaissRetriever(Retriever):
    """Retriever based on Faiss"""

    encoder: Param[TextEncoder]
    """The query encoder"""

    index: Param[BaseFaissIndex]
    """The faiss index"""

    topk: Param[int]
    """the number of documents to be retrieved"""

    @initializer
    def initialize(self):
        logger.info("FAISS retriever: initializing the encoder")
        self.encoder.initialize()

    def retrieve(self, query: str) -> List[ScoredDocument]:
        """Retrieves a documents, returning a list sorted by decreasing score"""
        with torch.no_grad():
            self.encoder.eval()  # pass the model to the evaluation model
            encoded_query = self.encoder([query])
            if self.index.normalize:
                encoded_query /= encoded_query.norm(2)

            values, indices = self.index.get_index().search(
                encoded_query.cpu().numpy(), self.topk
            )
            return [
                ScoredDocument(
                    self.index.documents.docid_internal2external(int(ix)), float(value)
                )
                for ix, value in zip(indices[0], values[0])
                if ix >= 0
            ]
