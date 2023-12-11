Pointwise
=========

Trainers are responsible for defining the the way to train
a learnable scorer.

.. autoxpmconfig:: xpmir.learning.trainers.multiple.MultipleTrainer

.. autoxpmconfig:: xpmir.letor.trainers.LossTrainer
   :members: process_microbatch

.. autoxpmconfig:: xpmir.letor.trainers.pointwise.PointwiseTrainer


.. currentmodule:: xpmir.letor.trainers.pointwise

Trainer
-------

.. autoxpmconfig:: PointwiseTrainer

Losses
------

.. autoxpmconfig:: PointwiseLoss
.. autoxpmconfig:: MSELoss
.. autoxpmconfig:: BinaryCrossEntropyLoss

Sampler
-------

.. currentmodule:: xpmir.letor.samplers

.. autoxpmconfig:: PointwiseSampler
    :members: pointwise_iter

.. autoxpmconfig:: PointwiseModelBasedSampler
