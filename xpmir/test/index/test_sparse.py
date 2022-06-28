from experimaestro.xpmutils import DirectoryContext
from pathlib import Path
import pytest
import torch
import numpy as np
from tqdm import tqdm
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.test.utils import SampleAdhocDocumentStore, SparseRandomTextEncoder


@pytest.fixture
def context(tmp_path: Path):
    from experimaestro.taskglobals import Env

    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir()
    return DirectoryContext(tmp_path)


class SparseIndex:
    def __init__(self, context):
        # Build the FAISS index
        documents = SampleAdhocDocumentStore(num_docs=100)
        self.encoder = SparseRandomTextEncoder(dim=1000, sparsity=0.7)
        builder = SparseRetrieverIndexBuilder(
            encoder=self.encoder, documents=documents, max_postings=10, batch_size=5
        )
        builder_instance = builder.instance(context=context)
        builder_instance.execute()

        self.document_store = builder_instance.documents
        self.x_docs = builder_instance.encoder(
            [d.text for d in self.document_store.documents.values()]
        )

        # Check index
        self.index = builder.config()
        self.index_instance = self.index.instance()
        self.topk = 10


@pytest.fixture
def sparse_index(context):
    return SparseIndex(context)


def test_sparse_indexation(sparse_index: SparseIndex):
    chosen_ix = np.random.choice(np.arange(len(sparse_index.x_docs.T)), 10)
    for ix in chosen_ix:
        x = sparse_index.x_docs[:, ix]
        nz = torch.nonzero(x)
        sparse_index.index_instance.initialize(False)
        it = sparse_index.index_instance.postings(0, ix)
        jx = 0
        while it.has_next():
            posting = it.next()
            assert (
                posting.docid == nz[jx]
            ), f"Error for posting {jx} of term {ix} (docid)"
            assert (
                x[nz[jx]] == posting.value
            ), f"Error for posting {jx} of term {ix} (value)"

            jx += 1


@pytest.fixture(params=[True, False])
def retriever(context, sparse_index: SparseIndex, request: bool):
    retriever = SparseRetriever(
        encoder=sparse_index.encoder,
        topk=10,
        batchsize=32,
        index=sparse_index.index,
        in_memory=request.param,
    ).instance(context=context)
    retriever.initialize()

    return retriever


def test_sparse_retrieve(sparse_index: SparseIndex, retriever):
    # Retrieve with the index
    scores = sparse_index.x_docs @ sparse_index.x_docs.T
    chosen_ix = np.random.choice(np.arange(len(sparse_index.x_docs)), 10)
    for ix in chosen_ix:
        document = sparse_index.document_store.document(ix)
        # , document in tqdm(enumerate(document_store.documents.values()), desc="Checking scores"):

        scoredDocuments = retriever.retrieve(document.text)
        scoredDocuments.sort(reverse=True)

        sorted = scores[ix].sort(descending=True)

        expected = list(sorted.indices[: retriever.topk].numpy())
        observed = [int(sd.docid) for sd in scoredDocuments]
        assert expected == observed

        expected_scores = sorted.values[: retriever.topk].numpy()
        observed_scores = np.array([float(sd.score) for sd in scoredDocuments])
        np.testing.assert_allclose(expected_scores, observed_scores, 1e-5)


def test_sparse_retrieve_all(retriever):
    queries = {"q1": "Query 1", "q2": "Query 2"}
    all_results = retriever.retrieve_all(queries)

    for key, query in queries.items():
        query_results = retriever.retrieve(query)
        assert [d.docid for d in all_results[key]] == [d.docid for d in query_results]
        assert [d.score for d in all_results[key]] == [d.score for d in query_results]
