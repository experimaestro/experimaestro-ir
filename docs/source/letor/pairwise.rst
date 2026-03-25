Pairwise
========

Trainer
-------

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseTrainer

Samplers
********

.. currentmodule:: xpmir.letor.samplers

.. autoxpmconfig:: PairwiseModelBasedSampler

.. autoxpmconfig:: TripletBasedSampler
.. autoxpmconfig:: PairwiseDatasetTripletBasedSampler
.. autoxpmconfig:: PairwiseInBatchNegativesSampler
.. autoxpmconfig:: PairwiseSamplerFromTSV
.. autoxpmconfig:: ModelBasedHardNegativeSampler

Dataset types
*************

.. autoxpmconfig:: xpmir.letor.samplers.JSONLPairwiseSampleDataset
.. autoxpmconfig:: xpmir.letor.samplers.TSVPairwiseSampleDataset

Adapters
********

.. autoxpmconfig:: xpmir.letor.samplers.adapters.SamplerAdapter

Processors
**********

.. autoxpmconfig:: xpmir.letor.processors.StoreHydrator
