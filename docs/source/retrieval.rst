Retrieval
=========

This page describes the components used for retrieval, i.e. finding the top
:math:`k` documents from a collection given a query. XPMIR supports classical
models (BM25, query-likelihood), dense retrieval (FAISS), sparse learned
retrieval, and late-interaction models (ColBERT/PLAID), as well as multi-stage
pipelines that combine them.

.. contents:: On this page
   :local:
   :depth: 2


Base classes
------------

Core data structures and the abstract retriever interface that all
implementations extend.

.. autoclass:: xpmir.rankers.ScoredDocument

.. autoxpmconfig:: xpmir.rankers.retriever.Retriever
    :members: initialize, collection, retrieve_all, retrieve

Standard IR models
------------------

Definitions for classical probabilistic retrieval models. These are
backend-agnostic specifications that can be instantiated with a concrete
engine such as :class:`~xpmir.interfaces.anserini.AnseriniRetriever`.

.. autoxpmconfig:: xpmir.rankers.standard.Model
.. autoxpmconfig:: xpmir.rankers.standard.BM25
.. autoxpmconfig:: xpmir.rankers.standard.QLDirichlet

Multi-stage retrievers
----------------------

In a re-ranking setting, a two-stage retriever first retrieves candidates with
a fast first-stage model, then re-scores them with a more expensive scorer.

The re-ranking process is memory-efficient: it uses lazy evaluation of
first-stage results and maximises GPU throughput by batching query-document
pairs across multiple queries.

.. autoxpmconfig:: xpmir.rankers.scorer.AbstractTwoStageRetriever
.. autoxpmconfig:: xpmir.rankers.scorer.TwoStageRetriever

Duo-retrievers
--------------

Duo-retrievers predict which of two candidate documents is more relevant to the
query (pairwise preference), rather than assigning an absolute score.

.. autoxpmconfig:: xpmir.rankers.scorer.DuoTwoStageRetriever
.. autoxpmconfig:: xpmir.rankers.scorer.DuoLearnableScorer

Miscellaneous retrievers
------------------------

Utility retrievers for loading pre-computed runs, hydrating results with
document text, or exhaustive scoring.

.. autoxpmconfig:: xpmir.rankers.full.FullRetriever
.. autoxpmconfig:: xpmir.rankers.full.FullRetrieverRescorer
.. autoxpmconfig:: xpmir.rankers.retriever.RetrieverHydrator
.. autoxpmconfig:: xpmir.rankers.retriever.RunRetriever


Distributed Retrieval
---------------------

XPMIR supports distributed retrieval and re-ranking across multiple GPUs using
`Lightning Fabric <https://lightning.ai/docs/fabric/>`_. **Currently, this
optimized distributed logic is implemented for :class:`~xpmir.index.sparse.SparseRetriever`
and :class:`~xpmir.rankers.scorer.TwoStageRetriever`.**

This is particularly useful for large-scale evaluation on thousands of queries.

How it works
^^^^^^^^^^^^

When a retriever is configured with a Fabric instance, the :meth:`~xpmir.rankers.retriever.Retriever.retrieve_all`
method leverages all available devices:

1. **Query Sharding**: The set of queries is automatically partitioned across
   the available GPUs.
2. **Parallel Processing**: Each device processes its assigned shard.
   - For **Sparse Retrieval** (:class:`~xpmir.index.sparse.SparseRetriever`),
     queries are encoded in batches and searched via asynchronous workers.
   - For **Two-Stage Retrieval** (:class:`~xpmir.rankers.scorer.TwoStageRetriever`),
     document re-ranking is batched across queries for maximum throughput.
3. **Result Gathering**: Once processing is complete, results are collected
   from all ranks and merged on the global zero rank.

Usage
^^^^^

Distributed retrieval is automatically enabled when using the :class:`~xpmir.evaluation.Evaluate`
task with a multi-GPU :class:`~xpm_torch.configuration.FabricConfiguration`.

To use it manually in a script:

.. code-block:: python

    from lightning import Fabric
    fabric = Fabric(devices=2, strategy="ddp")
    fabric.launch()

    retriever.initialize()
    retriever.setup_with_fabric(fabric)

    # Distributed retrieval
    results = retriever.retrieve_all(queries)

    # Results are gathered on rank 0
    if fabric.is_global_zero:
        print(f"Total queries retrieved: {len(results)}")


Index backends
--------------

The sections below describe the available index backends and their associated
retrievers.

Anserini
^^^^^^^^

`Anserini <https://github.com/castorini/anserini>`_ provides classical
inverted-index retrieval (BM25, query-likelihood, etc.) via Lucene.

.. autoxpmconfig:: xpmir.index.anserini.Index
.. autoxpmconfig:: xpmir.interfaces.anserini.AnseriniRetriever
.. autoxpmconfig:: xpmir.interfaces.anserini.IndexCollection
.. autoxpmconfig:: xpmir.interfaces.anserini.SearchCollection

FAISS
^^^^^

`FAISS <https://github.com/facebookresearch/faiss>`_ provides approximate
nearest-neighbour search for dense vector retrieval.

.. autoxpmconfig:: xpmir.index.faiss.FaissIndex
.. autoxpmconfig:: xpmir.index.faiss.IndexBackedFaiss
.. autoxpmconfig:: xpmir.index.faiss.FaissRetriever


fast-plaid (ColBERT / PLAID)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Interface to `fast-plaid <https://github.com/lightonai/fast-plaid>`_, a
Rust-based implementation of PLAID / ColBERT late-interaction retrieval.
Per-document token vectors can be reconstructed from the compressed index
via :meth:`~xpmir.index.plaid.PlaidIndex.get_document_tokens`.

.. autoxpmconfig:: xpmir.index.plaid.PlaidIndex
    :members: get_document_tokens
.. autoxpmconfig:: xpmir.index.plaid.PlaidIndexBuilder
.. autoxpmconfig:: xpmir.index.plaid.PlaidRetriever


Sparse retrieval
^^^^^^^^^^^^^^^^

Learned sparse retrieval indexes (e.g. for SPLADE), backed by the
`impact-index <https://github.com/experimaestro/impact-index>`_ Rust library.

.. autoxpmconfig:: xpmir.index.sparse.AbstractSparseRetrieverIndex
.. autoxpmconfig:: xpmir.index.sparse.AbstractSparseRetrieverIndexBuilder
.. autoxpmconfig:: xpmir.index.sparse.SparseRetriever

**Impact library (Rust)**

.. autoxpmconfig:: xpmir.index.sparse.SparseRetrieverIndex
.. autoxpmconfig:: xpmir.index.sparse.SparseRetrieverIndexBuilder

**Block-Max Pruning**

Adapters for `Faster Learned Sparse Retrieval with Block-Max Pruning
<https://arxiv.org/abs/2405.01117>`_.

.. autoxpmconfig:: xpmir.index.sparse.BMPSparseRetrieverIndex
.. autoxpmconfig:: xpmir.index.sparse.BMPSparseRetrieverIndexBuilder
.. autoxpmconfig:: xpmir.index.sparse.BMPSparseRetriever
