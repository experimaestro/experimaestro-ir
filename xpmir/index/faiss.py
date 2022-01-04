"""Interface to the Facebook FAISS library

https://github.com/facebookresearch/faiss
"""

from pathlib import Path
from typing import List
from experimaestro.core.objects import Config
import torch
from experimaestro import Annotated, Meta, Task, pathgenerator, Param, tqdm
import logging
from datamaestro_text.data.ir import AdhocDocumentStore
from xpmir.rankers import Retriever, ScoredDocument
from xpmir.letor.batchers import Batcher
from xpmir.vocab.encoders import TextEncoder
from xpmir.letor import Device, DEFAULT_DEVICE
from xpmir.utils import batchiter, easylog

logger = easylog()

try:
    import faiss
except ModuleNotFoundError:
    logging.error("FAISS library is not available (install faiss-cpu or faiss)")
    raise


class FaissIndex(Config):
    normalize: Param[bool]
    faiss_index: Annotated[Path, pathgenerator("faiss.dat")]
    documents: Param[AdhocDocumentStore]


class IndexBackedFaiss(FaissIndex, Task):
    """Constructs a FAISS index backed up by an index

    Attributes:

    index: The index that contains the raw documents and their ids
    encoder: The text encoder for documents
    batchsize: How many documents are processed at once

    """

    encoder: Param[TextEncoder]
    batchsize: Meta[int] = 1
    device: Meta[Device] = DEFAULT_DEVICE
    batcher: Meta[Batcher] = Batcher()

    def execute(self):
        self.encoder.initialize()
        index = faiss.IndexFlatL2(self.encoder.dimension)

        doc_iter = tqdm(
            self.documents.iter_documents(), total=self.documents.documentcount
        )
        batcher = self.batcher.initialize(self.batchsize)

        self.encoder.to(self.device(logger)).eval()
        with torch.no_grad():
            for batch in batchiter(self.batchsize, doc_iter, index):
                batcher.process(batch, self.index_documents)

        logging.info("Writing FAISS index (%d documents)", index.ntotal)
        faiss.write_index(index, str(self.faiss_index))

    def index_documents(self, batch: List[str], index):
        x = self.encoder(batch)
        if self.normalize:
            x /= x.norm(2, keepdim=True, dim=1)
        index.add(x.cpu().numpy())

    def docid_internal2external(self, docid: int):
        return self.documents.docid_internal2external(docid)


class FaissRetriever(Retriever):
    """Retriever based on Faiss

    Attributes:

    encoder: The query encoder
    index: The FAISS index
    """

    encoder: Param[TextEncoder]
    index: Param[FaissIndex]
    topk: Param[int]

    def __postinit__(self):
        self._index = faiss.read_index(str(self.index.faiss_index))

    def retrieve(self, query: str) -> List[ScoredDocument]:
        """Retrieves a documents, returning a list sorted by decreasing score"""
        with torch.no_grad():
            encoded_query = self.encoder([query])
            if self.index.normalize:
                encoded_query /= encoded_query.norm(2)

            distances, indices = self._index.search(
                encoded_query.cpu().numpy(), self.topk
            )
            return [
                ScoredDocument(
                    self.index.documents.docid_internal2external(ix), -distance
                )
                for ix, distance in zip(indices[0], distances[0])
            ]
