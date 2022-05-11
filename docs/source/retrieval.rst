Retrieval
=========

Models
------

.. autoxpmconfig:: xpmir.rankers.Retriever

Standard IR models
------------------

Standard IR models are definitions that can be used by a specific instance,
like e.g. :class:`xpmir.interfaces.anserini.AnseriniRetriever`

.. autoxpmconfig:: xpmir.rankers.standard.Model
.. autoxpmconfig:: xpmir.rankers.standard.BM25

Other retrievers
----------------

.. autoxpmconfig:: xpmir.rankers.TwoStageRetriever


Anserini
--------

.. autoxpmconfig:: xpmir.interfaces.anserini.Index
.. autoxpmconfig:: xpmir.interfaces.anserini.AnseriniRetriever
