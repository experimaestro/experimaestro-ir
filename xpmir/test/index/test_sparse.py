from experimaestro.xpmutils import DirectoryContext
from pathlib import Path
import torch
import numpy as np
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.test.index.utils import SampleAdhocDocumentStore
from xpmir.test.index.utils import SparseRandomTextEncoder


def test_sparse_indexation(tmp_path: Path):
    # Experimaestro context
    from experimaestro.taskglobals import Env

    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir()
    context = DirectoryContext(tmp_path)

    # Build the FAISS index
    documents = SampleAdhocDocumentStore(num_docs=100)
    encoder = SparseRandomTextEncoder(dim=1000, sparsity=0.99)
    builder = SparseRetrieverIndexBuilder(
        encoder=encoder, documents=documents, max_postings=10, batch_size=5
    )
    builder_instance = builder.instance(context=context)
    builder_instance.execute()

    document_store = builder_instance.documents
    x_docs = builder_instance.encoder(
        [d.text for d in document_store.documents.values()]
    )

    # Check index
    index = builder.config()
    index_instance = index.instance()
    for ix, x in enumerate(x_docs.T):
        nz = torch.nonzero(x)
        for jx, posting in enumerate(index_instance.iter_postings(0, ix)):
            assert posting.docid == nz[jx]
            assert x[nz[jx]] == posting.value

    # Retrieve with the index
    topk = 10
    retriever = SparseRetriever(encoder=encoder, topk=topk, index=index).instance(
        context=context
    )
    retriever.initialize()
    scores = x_docs @ x_docs.T

    for ix, document in enumerate(document_store.documents.values()):
        scoredDocuments = retriever.retrieve(document.text)
        scoredDocuments.sort(reverse=True)

        sorted = scores[ix].sort(descending=True)

        expected = list(sorted.indices[:topk].numpy())
        observed = [int(sd.docid) for sd in scoredDocuments]
        assert expected == observed

        expected_scores = sorted.values[:topk].numpy()
        observed_scores = np.array([float(sd.score) for sd in scoredDocuments])
        np.testing.assert_allclose(expected_scores, observed_scores, 1e-5)
