from collections import defaultdict
from typing import Dict, Iterator, List, OrderedDict
from experimaestro.xpmutils import DirectoryContext
import pytest
from experimaestro import Param
from pathlib import Path
from datamaestro_text.data.ir import AdhocDocument, AdhocDocumentStore
import torch
from xpmir.index.faiss import (
    IndexBackedFaiss,
    FaissIndex,
    FaissRetriever,
)
from xpmir.documents.samplers import DocumentSampler, HeadDocumentSampler
from xpmir.letor.records import Document
from xpmir.text.encoders import TextEncoder
import logging


DOCUMENTS = OrderedDict(
    (document.docid, document)
    for document in [
        AdhocDocument("0", "the cat sat on the mat"),
        AdhocDocument("1", "the purple car"),
        AdhocDocument("2", "my little dog"),
        AdhocDocument("3", "the truck was on track"),
    ]
)


class SampleAdhocDocumentStore(AdhocDocumentStore):
    id: Param[str] = ""

    @property
    def documentcount(self):
        return len(DOCUMENTS)

    def document_text(self, docid: str) -> str:
        """Returns the text of the document given its id"""
        return DOCUMENTS[docid].text

    def iter_documents(self) -> Iterator[AdhocDocument]:
        return iter(DOCUMENTS.values())

    def docid_internal2external(self, docid: int):
        """Converts an internal collection ID (integer) to an external ID"""
        return str(docid)


class RandomTextEncoder(TextEncoder):
    DIMENSION = 13

    # A default dict to always return the same embeddings
    MAP: Dict[str, torch.Tensor] = defaultdict(
        lambda: torch.randn(RandomTextEncoder.DIMENSION)
    )

    def __init__(self):
        super().__init__()

    @property
    def dimension(self) -> int:
        return RandomTextEncoder.DIMENSION

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Returns a matrix encoding the provided texts"""
        return torch.cat([RandomTextEncoder.MAP[text].unsqueeze(0) for text in texts])


indexspecs = ["Flat", "HNSW"]


@pytest.mark.parametrize("indexspec", indexspecs)
def test_faiss_indexation(tmp_path: Path, indexspec):
    # Experimaestro context
    from experimaestro.taskglobals import Env

    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir()
    context = DirectoryContext(tmp_path)

    # Build the FAISS index
    documents = SampleAdhocDocumentStore()
    sampler = HeadDocumentSampler(documents=documents, max_ratio=0.5)
    encoder = RandomTextEncoder()
    builder = IndexBackedFaiss(
        indexspec=indexspec,
        encoder=encoder,
        normalize=False,
        sampler=sampler,
        documents=documents,
    )
    builder_instance = builder.instance(context=context)
    builder_instance.execute()

    # Retrieve with the index
    retriever = FaissRetriever(encoder=encoder, topk=10, index=builder).instance(
        context=context
    )
    retriever.initialize()
    x_docs = retriever.encoder([d.text for d in DOCUMENTS.values()])
    scores = x_docs @ x_docs.T

    for ix, document in enumerate(DOCUMENTS.values()):
        scoredDocuments = retriever.retrieve(document.text)
        scoredDocuments.sort(reverse=True)

        expected = list(scores[ix].sort().indices.numpy()[::-1])
        logging.warning("%s vs %s", scores[ix], scoredDocuments)
        observed = [int(sd.docid) for sd in scoredDocuments]

        assert expected == observed
