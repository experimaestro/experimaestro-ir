import logging
from pathlib import Path

import pytest
from experimaestro.xpmutils import DirectoryContext
from xpmir.documents.samplers import HeadDocumentSampler
from xpmir.index.faiss import FaissRetriever, IndexBackedFaiss
from xpmir.test.utils.utils import SampleAdhocDocumentStore, SparseRandomTextEncoder

indexspecs = ["Flat", "HNSW"]


@pytest.mark.parametrize("indexspec", indexspecs)
def test_faiss_indexation(tmp_path: Path, indexspec):
    # Experimaestro context
    from experimaestro.taskglobals import Env

    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir()
    context = DirectoryContext(tmp_path)

    # Build the FAISS index
    documents = SampleAdhocDocumentStore(num_docs=100)
    sampler = HeadDocumentSampler(documents=documents, max_ratio=0.5)
    encoder = SparseRandomTextEncoder(dim=1000, sparsity=0.0)
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
    topk = 10
    retriever = FaissRetriever(encoder=encoder, topk=topk, index=builder).instance(
        context=context
    )
    retriever.initialize()

    documents = builder_instance.documents.documents
    x_docs = retriever.encoder([d.text for d in documents.values()])
    scores = x_docs @ x_docs.T

    for ix, document in enumerate(documents.values()):
        scoredDocuments = retriever.retrieve(document.text)
        scoredDocuments.sort(reverse=True)

        expected = list(scores[ix].sort(descending=True).indices[:topk].numpy())
        logging.warning("%s vs %s", scores[ix], scoredDocuments)
        observed = [int(sd.docid) for sd in scoredDocuments]

        assert expected == observed
