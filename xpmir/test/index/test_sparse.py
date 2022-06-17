from experimaestro.xpmutils import DirectoryContext
from pathlib import Path
import logging
import torch
import numpy as np
from tqdm import tqdm
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
    encoder = SparseRandomTextEncoder(dim=1000, sparsity=0.7)
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

    chosen_ix = np.random.choice(np.arange(len(x_docs.T)), 10)
    for ix in chosen_ix:
        x = x_docs[:, ix]
        nz = torch.nonzero(x)
        for jx, posting in enumerate(index_instance.iter_postings(0, ix)):
            assert (
                posting.docid == nz[jx]
            ), f"Error for posting {jx} of term {ix} (docid)"
            assert (
                x[nz[jx]] == posting.value
            ), f"Error for posting {jx} of term {ix} (value)"

    # Retrieve with the index
    topk = 10
    retriever = SparseRetriever(encoder=encoder, topk=topk, index=index).instance(
        context=context
    )
    retriever.initialize()
    scores = x_docs @ x_docs.T

    chosen_ix = np.random.choice(np.arange(len(x_docs)), 10)
    for ix in chosen_ix:
        document = document_store.document(ix)
        # , document in tqdm(enumerate(document_store.documents.values()), desc="Checking scores"):
        scoredDocuments = retriever.retrieve(document.text)
        scoredDocuments.sort(reverse=True)

        sorted = scores[ix].sort(descending=True)

        expected = list(sorted.indices[:topk].numpy())
        observed = [int(sd.docid) for sd in scoredDocuments]
        assert expected == observed

        expected_scores = sorted.values[:topk].numpy()
        observed_scores = np.array([float(sd.score) for sd in scoredDocuments])
        np.testing.assert_allclose(expected_scores, observed_scores, 1e-5)
