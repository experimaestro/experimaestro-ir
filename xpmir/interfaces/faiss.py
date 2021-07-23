"""Interface to the Facebook FAISS library

https://github.com/facebookresearch/faiss
"""

from pathlib import Path
from typing import List
import torch
from experimaestro import Annotated, Option, Task, pathgenerator, Param, tqdm
import logging

from xpmir.index.base import Index
from xpmir.neural.siamese import TextEncoder
from xpmir.rankers import Retriever, ScoredDocument

try:
    import faiss
except ImportError:
    logging.error("FAISS library is not available (install faiss-cpu or faiss)")
    raise


class FaissIndex(Index):
    normalize: Param[bool]
    faiss_index: Annotated[Path, pathgenerator("faiss.dat")]


class IndexBackedFaiss(FaissIndex, Task):
    """Constructs a FAISS index backed up by an index

    Attributes:

    index: The index that contains the raw documents and their ids
    encoder: The text encoder for documents
    batchsize: How many documents are processed at once

    """

    index: Param[Index]
    encoder: Param[TextEncoder]
    batchsize: Option[int] = 1

    def execute(self):
        index = faiss.IndexFlatL2(self.encoder.dimension)

        def batches():
            batch = []
            for _, text in tqdm(
                self.index.iter_documents(), total=self.index.documentcount
            ):
                batch.append(text)
                if len(batch) >= self.batchsize:
                    yield batch
                    batch = []

            if batch:
                yield batch

        with torch.no_grad():
            for batch in batches():
                x = self.encoder(batch)
                if self.normalize:
                    x /= x.norm(2, keepdim=True, dim=1)
                index.add(x.cpu().numpy())

        logging.info("Writing FAISS index (%d documents)", index.ntotal)
        faiss.write_index(index, str(self.faiss_index))

    def docid_internal2external(self, docid: int):
        return self.index.docid_internal2external(docid)


class FaissRetriever(Retriever):
    """Retriever based on Faiss

    Attributes:

    encoder: The query encoder
    index: The FAISS index
    """

    encoder: Param[TextEncoder]
    index: Param[FaissIndex]

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
                ScoredDocument(self.index.docid_internal2external(ix), -distance)
                for ix, distance in zip(indices[0], distances[0])
            ]
