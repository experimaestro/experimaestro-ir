Pairwise
========

Trainers are responsible for defining the the way to train
a learnable scorer.

.. autoxpmconfig:: xpmir.learning.trainers.multiple.MultipleTrainer

.. autoxpmconfig:: xpmir.letor.trainers.LossTrainer
   :members: process_microbatch

.. autoxpmconfig:: xpmir.letor.trainers.pointwise.PointwiseTrainer

Trainer
-------

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseTrainer
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.DuoPairwiseTrainer
.. autoxpmconfig:: xpmir.letor.trainers.generative.GenerativeTrainer


Losses
------

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseLoss
   :members: compute

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.CrossEntropyLoss
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.HingeLoss
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PointwiseCrossEntropyLoss
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseLossWithTarget

Pairwise
--------

.. currentmodule:: xpmir.letor.samplers


Samplers
********

.. autoxpmconfig:: PairwiseSampler
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

Dataset transforms
******************

.. autoxpmconfig:: xpmir.letor.samplers.synthetic.JSONLNegativeGeneration

Adapters
********

.. autoxpmconfig:: xpmir.letor.samplers.hydrators.PairwiseTransformAdapter
