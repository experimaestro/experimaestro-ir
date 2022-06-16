Retrieval
=========

Models
------

.. autoclass:: xpmir.rankers.ScoredDocument

.. autoxpmconfig:: xpmir.rankers.Retriever
    :members: initialize, collection, getindex, retrieve_all, retrieve

Standard IR models
------------------

Standard IR models are definitions that can be used by a specific instance,
like e.g. :class:`xpmir.interfaces.anserini.AnseriniRetriever`

.. autoxpmconfig:: xpmir.rankers.standard.Model
.. autoxpmconfig:: xpmir.rankers.standard.BM25

Other retrievers
----------------

In a re-ranking setting, one can use a two stage retriever to perform
retrieval, by using a fully fledge retriever first, and then
re-ranking the results.

.. autoxpmconfig:: xpmir.rankers.TwoStageRetriever


Anserini
--------

.. autoxpmconfig:: xpmir.interfaces.anserini.Index
.. autoxpmconfig:: xpmir.interfaces.anserini.AnseriniRetriever
