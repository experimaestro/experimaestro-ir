from collections import OrderedDict, defaultdict
from typing import ClassVar, Dict, Iterator, List, Tuple
import torch
from attrs import define
import numpy as np
from datamaestro_text.data.ir import Document, DocumentStore
from datamaestro_text.data.ir.base import GenericDocument

from experimaestro import Param
from xpmir.text.encoders import TextEncoder


@define(frozen=True)
class GenericDocumentWithID(GenericDocument):
    internal_docid: int


class SampleDocumentStore(DocumentStore):
    id: Param[str] = ""
    num_docs: Param[int] = 200

    def __post_init__(self):
        # Generate all the documents
        self.documents = OrderedDict(
            (
                str(ix),
                GenericDocumentWithID(str(ix), f"Document {ix}", internal_docid=ix),
            )
            for ix in range(self.num_docs)
        )

    @property
    def documentcount(self):
        return len(self.documents)

    def document_int(self, internal_docid: int) -> Document:
        return self.documents[str(internal_docid)]

    def document_ext(self, docid: str) -> Document:
        """Returns the text of the document given its id"""
        return self.documents[docid]

    def iter_documents(self) -> Iterator[Document]:
        return iter(self.documents.values())

    def docid_internal2external(self, docid: int):
        """Converts an internal collection ID (integer) to an external ID"""
        return str(docid)

    def __iter__(self):
        return iter(self.documents.values())


class VectorGenerator:
    def __init__(self, dimension, sparsity):
        self.dimension = dimension
        self.num_zeros_rate = torch.FloatTensor([sparsity * self.dimension])

    def __call__(self) -> torch.Tensor:
        x = torch.randn(self.dimension)
        if self.num_zeros_rate.item() == 0:
            return x

        zeros = int(min(torch.poisson(self.num_zeros_rate).item(), self.dimension - 1))
        x[np.random.choice(np.arange(self.dimension), zeros)] = 0
        x = x.abs()
        return x


class SparseRandomTextEncoder(TextEncoder):
    # A default dict to always return the same embeddings
    MAPS: ClassVar[Dict[Tuple[int, float], Dict[str, torch.Tensor]]] = {}

    map: Dict[str, torch.Tensor]
    dim: Param[int]
    sparsity: Param[float] = 0.0

    def __post_init__(self):
        super().__init__()
        if not (self.dim, self.sparsity) in SparseRandomTextEncoder.MAPS:
            SparseRandomTextEncoder.MAPS[(self.dim, self.sparsity)] = defaultdict(
                VectorGenerator(self.dim, self.sparsity)
            )

        # Get the given map
        self.map = SparseRandomTextEncoder.MAPS[(self.dim, self.sparsity)]

    @property
    def dimension(self):
        return self.dim

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a matrix encoding the provided texts"""
        return torch.cat([self.map[text].unsqueeze(0) for text in texts])
