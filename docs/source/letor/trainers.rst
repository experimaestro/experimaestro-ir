Trainers
========

Trainers are responsible for defining the the way to train
a learnable scorer.

.. autoxpmconfig:: xpmir.letor.trainers.Trainer

.. autoxpmconfig:: xpmir.letor.trainers.LossTrainer
   :members: process_microbatch

.. autoxpmconfig:: xpmir.letor.trainers.pointwise.PointwiseTrainer

Pointwise
********

.. currentmodule:: xpmir.letor.trainers.pointwise

Trainer
-------

.. autoxpmconfig:: PointwiseTrainer

Losses
------

.. autoxpmconfig:: PointwiseLoss
.. autoxpmconfig:: MSELoss
.. autoxpmconfig:: BinaryCrossEntropyLoss


Pairwise
********

Trainer
-------

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseTrainer

Losses
------

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseLoss
   :members: compute

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.CrossEntropyLoss
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.SoftmaxLoss
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.LogSoftmaxLoss
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.HingeLoss

Other
*****

.. autoxpmconfig:: xpmir.letor.trainers.multiple.MultipleTrainer


Distillation Trainers

=====================

Pairwise
********

Trainer
-------

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationPairwiseTrainer

Losses
------

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationPairwiseLoss
   :members: compute

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.MSEDifferenceLoss
.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationKLLoss
