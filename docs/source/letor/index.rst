Learning to rank
----------------

.. toctree::
   :maxdepth: 1

   pointwise
   pairwise
   batchwise
   distillation
   generative
   mlm
   generation
   alignment


Learning to rank is handled by various classes. Some are located
in the :ref:`learning module <Learning>`.

Listeners
=========

.. autoxpmconfig:: xpmir.letor.learner.ValidationListener
.. autoxpmconfig:: xpmir.letor.learner.ValidationModuleLoader

Scorers
=======

Scorers are able to give a score to a (query, document) pair. Among the
scorers, some are have learnable parameters.

.. autoxpmconfig:: xpmir.rankers.Scorer
   :members: initialize, rsv, to, eval, getRetriever
.. autoxpmconfig:: xpmir.rankers.RandomScorer
.. autoxpmconfig:: xpmir.rankers.AbstractModuleScorer
.. autoxpmconfig:: xpmir.rankers.LearnableScorer

Adapters
********

.. autoxpmconfig:: xpmir.rankers.adapters.ScorerTransformAdapter

Utility functions
*****************

.. autofunction:: xpmir.rankers.scorer_retriever


Retrievers
==========

Scores can be used as retrievers through a :py:class:`xpmir.rankers.TwoStageRetriever`

Samplers
--------

.. currentmodule:: xpmir.letor.samplers

Samplers provide samples in the form of *records*. They all inherit from:

.. autoclass:: SerializableIterator


.. autoxpmconfig:: ModelBasedSampler


Records for training
--------------------

.. automodule:: xpmir.letor.records
   :members: PointwiseRecord, PairwiseRecord


Document samplers
=================

Useful for pre-training or when learning index parameters (e.g. for FAISS).

.. currentmodule:: xpmir.documents.samplers
.. autoxpmconfig:: DocumentSampler
.. autoxpmconfig:: HeadDocumentSampler
.. autoxpmconfig:: RandomDocumentSampler

Adapters
********

.. autoxpmconfig:: xpmir.letor.samplers.hydrators.SampleTransform
.. autoxpmconfig:: xpmir.letor.samplers.hydrators.SampleHydrator
.. autoxpmconfig:: xpmir.letor.samplers.hydrators.SamplePrefixAdding
.. autoxpmconfig:: xpmir.letor.samplers.hydrators.SampleTransformList
