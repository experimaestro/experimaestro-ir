Retrieval
=========

This page describes the different configurations/tasks needed for retrieval, i.e. searching
the a subset of :math:`k` documents given a query.


- `Base class`_ shows the main class used for retrieval,
- `Standard IR models`_ describes the configurations for standard IR models like BM25,
- `Multi-stage retrievers` describes the configurations handling multi-stage retrieval (e.g. two-stage retriever)
- `Factories`_ describe utility classes and decorators that can be used to build retrievers that depend on a dataset.

Finally, retrieval interfaces to other libraries are given for `Anserini`_, `FAISS`_.



Base class
----------

.. autoclass:: xpmir.rankers.ScoredDocument

.. autoxpmconfig:: xpmir.rankers.Retriever
    :members: initialize, collection, getindex, retrieve_all, retrieve

Standard IR models
------------------

Standard IR models are definitions that can be used by a specific instance,
like e.g. :class:`xpmir.interfaces.anserini.AnseriniRetriever`

.. autoxpmconfig:: xpmir.rankers.standard.Model
.. autoxpmconfig:: xpmir.rankers.standard.BM25
.. autoxpmconfig:: xpmir.rankers.standard.QLDirichlet

Multi-stage retrievers
----------------------

In a re-ranking setting, one can use a two stage retriever to perform
retrieval, by using a fully fledge retriever first, and then
re-ranking the results.

.. autoxpmconfig:: xpmir.rankers.AbstractTwoStageRetriever
.. autoxpmconfig:: xpmir.rankers.TwoStageRetriever

Duo-retrievers
--------------

Duo-retrievers only predicts whether a document is "more relevant" than
another

.. autoxpmconfig:: xpmir.rankers.DuoTwoStageRetriever
.. autoxpmconfig:: xpmir.rankers.DuoLearnableScorer

Misc
----

.. autoxpmconfig:: xpmir.rankers.full.FullRetriever
.. autoxpmconfig:: xpmir.rankers.full.FullRetrieverRescorer
.. autoxpmconfig:: xpmir.rankers.RetrieverHydrator
.. autoxpmconfig:: xpmir.rankers.mergers.SumRetriever

Collection dependendant
-----------------------




Anserini
--------

.. autoxpmconfig:: xpmir.index.anserini.Index
.. autoxpmconfig:: xpmir.interfaces.anserini.Index
.. autoxpmconfig:: xpmir.interfaces.anserini.AnseriniRetriever
.. autoxpmconfig:: xpmir.interfaces.anserini.IndexCollection
.. autoxpmconfig:: xpmir.interfaces.anserini.SearchCollection

FAISS
-----

.. autoxpmconfig:: xpmir.index.faiss.FaissIndex
.. autoxpmconfig:: xpmir.index.faiss.IndexBackedFaiss
.. autoxpmconfig:: xpmir.index.faiss.FaissRetriever


Sparse
------

.. autoxpmconfig:: xpmir.index.sparse.SparseRetriever
.. autoxpmconfig:: xpmir.index.sparse.SparseRetrieverIndex
.. autoxpmconfig:: xpmir.index.sparse.SparseRetrieverIndexBuilder
