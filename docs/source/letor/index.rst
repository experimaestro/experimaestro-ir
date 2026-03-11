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

Naming Conventions
==================

To maintain architectural clarity, the project uses the following naming conventions for data objects:

*   **Records**: Low-level data structures (e.g., `IDTextRecord`, `ScoreRecord`). These are implemented as `TypedDict` and represent raw data or identifiers.
*   **Samples**: Data-layer objects (e.g., `PairwiseSample`). These are typically found in the data layer (datamaestro) and represent "raw" data containers, often containing multiple candidates or non-hydrated information.
*   **Items**: Model-ready objects (e.g., `PointwiseItem`, `PairwiseItem`). These are hydrated classes used in the model layer (xpmir), ready to be converted into tensors for training or inference.


Samplers
--------

.. currentmodule:: xpmir.letor.samplers

Samplers provide model-ready **items**. They all inherit from:

.. autoclass:: SerializableIterator


.. autoxpmconfig:: ModelBasedSampler


Items for training
------------------

.. automodule:: xpmir.letor.records
   :members: PointwiseItem, PairwiseItem, ListwiseItem, BatchwiseItems



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
