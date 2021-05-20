"""Interface to the Facebook FAISS library

https://github.com/facebookresearch/faiss
"""

from pathlib import Path
from experimaestro import Annotated, Config, Task, pathgenerator, Param
import logging

from xpmir.dm.data.base import Index
from xpmir.neural.siamese import TextEncoder

try:
    import faiss
except ImportError:
    logging.error("FAISS library is not available (install faiss-cpu or faiss)")
    raise


class FaissIndex(Config):
    pass


class IndexBackedFaiss(Task):
    """Constructs an index backed up by an index

    Attributes:

    index: The index that contains the raw documents and their ids
    encoder: The text encoder

    """

    index: Param[Index]
    encoder: Param[TextEncoder]
    faiss_index: Annotated[Path, pathgenerator("faiss.dat")]

    def execute(self):
        index = faiss.IndexFlatL2(self.encoder.dimension)

        for docid, text in index.documents():
            x = encoder(text)
            index.add(x)

        faiss.write_index(index, str(self.faiss_index))


class FaissRetriever:
    """Retriever based on Faiss"""

    encoder: Param[TextEncoder]
