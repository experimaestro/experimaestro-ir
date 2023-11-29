Pairwise
========

Trainers are responsible for defining the the way to train
a learnable scorer.

.. autoxpmconfig:: xpmir.letor.trainers.Trainer
.. autoxpmconfig:: xpmir.learning.trainers.multiple.MultipleTrainer

.. autoxpmconfig:: xpmir.letor.trainers.LossTrainer
   :members: process_microbatch

.. autoxpmconfig:: xpmir.letor.trainers.pointwise.PointwiseTrainer

Trainer
-------

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseTrainer

Losses
------

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseLoss
   :members: compute

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.CrossEntropyLoss
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.HingeLoss
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PointwiseCrossEntropyLoss


Pairwise
--------

.. autoxpmconfig:: PairwiseSampler
.. autoxpmconfig:: BatchwiseSampler
.. autoxpmconfig:: PairwiseModelBasedSampler
.. autoxpmconfig:: xpmir.documents.samplers.BatchwiseRandomSpanSampler

.. autoxpmconfig:: TripletBasedSampler
.. autoxpmconfig:: PairwiseDatasetTripletBasedSampler
.. autoxpmconfig:: PairwiseInBatchNegativesSampler
.. autoxpmconfig:: PairwiseSampleDatasetFromTSV
.. autoxpmconfig:: PairwiseSamplerFromTSV
.. autoxpmconfig:: ModelBasedHardNegativeSampler
