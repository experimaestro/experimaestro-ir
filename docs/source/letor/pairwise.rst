Pairwise learning
=================

In pairwise learning, each training instance is a (query, positive document,
negative document) triplet. The model learns to rank the positive document
above the negative one, typically optimised with a margin-based or
cross-entropy loss.

Trainer
-------

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseTrainer

Samplers
--------

Samplers produce pairwise training triplets from different data sources (model
scores, pre-computed files, or in-batch negatives).

.. currentmodule:: xpmir.letor.samplers

.. autoxpmconfig:: PairwiseModelBasedSampler

.. autoxpmconfig:: TripletBasedSampler
.. autoxpmconfig:: PairwiseDatasetTripletBasedSampler
.. autoxpmconfig:: PairwiseInBatchNegativesSampler
.. autoxpmconfig:: PairwiseSamplerFromTSV
.. autoxpmconfig:: ModelBasedHardNegativeSampler

Dataset types
-------------

Pre-computed pairwise datasets stored as JSONL or TSV files.

.. autoxpmconfig:: xpmir.letor.samplers.JSONLPairwiseSampleDataset
.. autoxpmconfig:: xpmir.letor.samplers.TSVPairwiseSampleDataset

Adapters
--------

.. autoxpmconfig:: xpmir.letor.samplers.adapters.SamplerAdapter

Processors
----------

.. autoxpmconfig:: xpmir.letor.processors.StoreHydrator
