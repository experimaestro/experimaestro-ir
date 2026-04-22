Learning to Rank
================

This section covers the learning-to-rank (LTR) components: scorers that assign
relevance scores to query-document pairs, samplers that produce training data,
and trainers that optimise model parameters. XPMIR supports pointwise, pairwise,
batchwise, and distillation training strategies.

.. toctree::
   :maxdepth: 1

   pointwise
   pairwise
   batchwise
   distillation


Scorers
-------

Scorers assign a relevance score to a (query, document) pair.
:class:`~xpmir.rankers.scorer.AbstractModuleScorer` is the base class for
scorers with learnable parameters (neural models).

.. autoxpmconfig:: xpmir.rankers.scorer.Scorer
   :members: initialize, rsv, compute, getRetriever
.. autoxpmconfig:: xpmir.rankers.scorer.RandomScorer
.. autoxpmconfig:: xpmir.rankers.scorer.AbstractModuleScorer

.. autofunction:: xpmir.rankers.scorer_retriever


Retrievers from scorers
-----------------------

Scorers can be wrapped as retrievers through a
:class:`~xpmir.rankers.scorer.TwoStageRetriever` (see :doc:`/retrieval`).


Naming conventions
------------------

The project uses consistent naming for data objects at different layers:

*   **Records** -- Low-level data structures (e.g. ``IDTextRecord``,
    ``ScoreRecord``). Implemented as ``TypedDict`` for raw data or identifiers.
*   **Samples** -- Data-layer objects (e.g. ``PairwiseSample``). Found in
    datamaestro; represent raw containers, possibly non-hydrated.
*   **Items** -- Model-ready objects (e.g. ``PointwiseItem``, ``PairwiseItem``).
    Hydrated objects used in the training loop, ready to be converted into
    tensors.


Samplers
--------

Samplers generate model-ready **items** from a dataset and a scorer (used for
hard-negative mining or scoring).

.. currentmodule:: xpmir.letor.samplers

.. autoxpmconfig:: ModelBasedSampler


Training items
--------------

Data classes representing training instances at different granularities.

.. automodule:: xpmir.letor.records
   :members: PointwiseItem, PairwiseItem, ListwiseItem, BatchwiseItems


Document samplers
-----------------

Samplers that produce documents (without queries). Useful for pre-training
objectives or for learning index parameters (e.g. FAISS quantisers).

.. currentmodule:: xpmir.documents.samplers
.. autoxpmconfig:: DocumentSampler
.. autoxpmconfig:: HeadDocumentSampler
.. autoxpmconfig:: RandomDocumentSampler

Sample adapters
---------------

Transforms applied to samples before they reach the model (e.g. hydrating
document text from a store, adding query prefixes).

.. autoxpmconfig:: xpmir.letor.samplers.hydrators.SampleTransform
.. autoxpmconfig:: xpmir.letor.samplers.hydrators.SampleHydrator
.. autoxpmconfig:: xpmir.letor.samplers.hydrators.SamplePrefixAdding
.. autoxpmconfig:: xpmir.letor.samplers.hydrators.SampleTransformList
