from experimaestro import ObjectStore
from experimaestro.xpmutils import DirectoryContext
from pathlib import Path
import pytest
import torch
import numpy as np
from xpmir.index.sparse import SparseRetriever, SparseRetrieverIndexBuilder
from xpmir.test.utils.utils import SampleDocumentStore, SparseRandomTextEncoder


@pytest.fixture
def context(tmp_path: Path):
    from experimaestro.taskglobals import Env

    Env.taskpath = tmp_path / "task"
    Env.taskpath.mkdir()
    return DirectoryContext(tmp_path)


class SparseIndex:
    def __init__(self, context, ordered_index: bool = False):
        # Build the index
        objects = ObjectStore()
        documents = SampleDocumentStore(num_docs=500)
        self.encoder = SparseRandomTextEncoder(dim=1000, sparsity=0.8)
        builder = SparseRetrieverIndexBuilder(
            encoder=self.encoder,
            documents=documents,
            max_postings=10,
            batch_size=5,
            ordered_index=ordered_index,
        )

        # Build the index
        builder_instance = builder.instance(context=context, objects=objects)
        builder_instance.execute()

        self.document_store = builder_instance.documents
        self.x_docs = builder_instance.encoder(
            [d.text for d in self.document_store.documents.values()]
        )

        # Check index
        self.index = builder.task_outputs(lambda x: x)
        self.index_instance = self.index.instance(objects=objects)
        self.topk = 10


@pytest.fixture(params=[False])
def sparse_index(context, request):
    return SparseIndex(context, ordered_index=request.param)


def test_sparse_indexation(sparse_index: SparseIndex):
    for ix in np.random.choice(np.arange(len(sparse_index.x_docs.T)), 10):
        # for ix in range(len(sparse_index.x_docs.T)):
        x = sparse_index.x_docs[:, ix]

        # nz indices are indices of documents
        nz = torch.nonzero(x)

        if sparse_index.index_instance.ordered:
            sorted_ix = sorted(range(len(nz)), key=lambda jx: -x[nz[jx]])
            nz = nz[sorted_ix]

        sparse_index.index_instance.initialize(False)
        for (jx, posting) in enumerate(sparse_index.index_instance.index.postings(ix)):
            assert (
                posting.docid == nz[jx]
            ), f"Error for posting {jx} of term {ix} (docid)"
            assert (
                x[nz[jx]] == posting.value
            ), f"Error for posting {jx} of term {ix} (value)"

            jx += 1


@pytest.fixture(params=[False])
def retriever(context, sparse_index: SparseIndex, request: bool):
    retriever = SparseRetriever(
        encoder=sparse_index.encoder,
        topk=sparse_index.topk,
        batchsize=2,
        index=sparse_index.index,
        in_memory=request.param,
    ).instance(context=context)
    retriever.initialize()

    return retriever


def test_sparse_retrieve(sparse_index: SparseIndex, retriever):
    # Computes the score directly
    x_docs = sparse_index.x_docs.type(torch.float32)

    # Choose a few documents
    chosen_ix = np.random.choice(np.arange(len(sparse_index.x_docs)), 10)
    for ix in chosen_ix:
        document = sparse_index.document_store.document_int(ix)

        # Use the retriever
        scoredDocuments = retriever.retrieve(document.get_text())
        # scoredDocuments.sort(reverse=True)
        # scoredDocuments = scoredDocuments[:retriever.topk]

        # Use the pre-computed scores
        scores = x_docs[ix] @ x_docs.T
        sorted = scores.sort(descending=True, stable=True)
        indices = sorted.indices[: retriever.topk]
        expected = list(indices.numpy())

        observed = [int(sd.document.get_id()) for sd in scoredDocuments]
        expected_scores = sorted.values[: retriever.topk].numpy()
        observed_scores = np.array([float(sd.score) for sd in scoredDocuments])

        np.testing.assert_allclose(
            expected_scores,
            observed_scores,
            1e-5,
            err_msg=f"{ix} {expected} vs {observed}",
        )
        assert expected == observed


def test_sparse_retrieve_all(retriever):
    """Just verifies that the retriever is coherent with itself when retrieving
    many queries"""
    queries = {
        "q1": "Query 1",
        "q2": "Query 2",
        "q3": "Query 3",
        "q4": "Query 4",
        "q5": "Query 55",
    }
    all_results = retriever.retrieve_all(queries)

    for key, query in queries.items():
        query_results = retriever.retrieve(query)

        observed = [d.document.get_id() for d in all_results[key]]
        expected = [d.document.get_id() for d in query_results]
        assert observed == expected

        observed_scores = [d.score for d in all_results[key]]
        expected_scores = [d.score for d in query_results]
        np.testing.assert_allclose(
            expected_scores, observed_scores, 1e-5, err_msg=f"{expected} vs {observed}"
        )
