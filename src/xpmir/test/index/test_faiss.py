import logging
from pathlib import Path

import pytest
from experimaestro import ObjectStore
from experimaestro.xpmutils import DirectoryContext
from xpmir.documents.samplers import HeadDocumentSampler
from xpmir.index.faiss import FaissRetriever, IndexBackedFaiss
from xpmir.test.utils.utils import SampleDocumentStore, DenseRandomTextEncoder

indexspecs = ["Flat", "HNSW"]


@pytest.mark.parametrize("indexspec", indexspecs)
def test_faiss_indexation(tmp_path: Path, indexspec):
    # Experimaestro context
    from experimaestro.taskglobals import Env

    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir()
    context = DirectoryContext(tmp_path)
    objects = ObjectStore()

    # Build the FAISS index
    documents = SampleDocumentStore.C(num_docs=100)
    sampler = HeadDocumentSampler.C(documents=documents, max_ratio=0.5)
    encoder = DenseRandomTextEncoder.C(dim=1000)
    builder = IndexBackedFaiss.C(
        indexspec=indexspec,
        encoder=encoder,
        normalize=False,
        sampler=sampler,
        documents=documents,
    )
    builder_instance = builder.instance(context=context, objects=objects)
    builder_instance.execute()

    # Retrieve with the index
    topk = 10
    retriever = FaissRetriever.C(encoder=encoder, topk=topk, index=builder).instance(
        context=context, objects=objects
    )
    retriever.initialize()

    documents = builder_instance.documents.documents
    x_docs = retriever.encoder([d["text_item"].text for d in documents.values()]).value
    x_docs = retriever.encoder([d["text_item"].text for d in documents.values()]).value
    scores = x_docs @ x_docs.T

    for ix, document in enumerate(documents.values()):
        scoredDocuments = retriever.retrieve(document)
        scoredDocuments.sort(reverse=True)

        expected = list(
            int(s) for s in scores[ix].sort(descending=True).indices[:topk].numpy()
        )
        logging.debug("%s vs %s", scores[ix], scoredDocuments)
        observed = [int(sd.document["id"]) for sd in scoredDocuments]

        if indexspec == "Flat":
            assert expected == observed
        else:
            # Approximate indices (HNSW) may not return exact top-k
            overlap = len(set(expected) & set(observed))
            assert overlap >= topk - 1, (
                f"Expected at least {topk - 1}/{topk} overlap, got {overlap}. "
                f"Expected {expected}, observed {observed}"
            )
