Trainers
========

Trainers are responsible for defining the the way to train
a learnable scorer.

.. autoxpmconfig:: xpmir.letor.trainers.Trainer
.. autoxpmconfig:: xpmir.learning.trainers.multiple.MultipleTrainer

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
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.HingeLoss
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PointwiseCrossEntropyLoss


Pairwise (duo)
**************

Trainer
-------

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseLossWithTarget
   :members: compute

Losses
------

.. autoxpmconfig:: xpmir.letor.trainers.pairwise.DuoPairwiseTrainer


Batchwise
*********

Trainer
-------

.. autoxpmconfig:: xpmir.letor.trainers.batchwise.BatchwiseTrainer

Losses
------

.. autoxpmconfig:: xpmir.letor.trainers.batchwise.BatchwiseLoss
.. autoxpmconfig:: xpmir.letor.trainers.batchwise.CrossEntropyLoss
.. autoxpmconfig:: xpmir.letor.trainers.batchwise.SoftmaxCrossEntropy


Distillation: Pairwise
**********************


Sampler
-------

.. autoxpmconfig:: xpmir.letor.distillation.samplers.DistillationPairwiseSampler
.. autoxpmconfig:: xpmir.letor.distillation.samplers.PairwiseHydrator

Trainer
-------

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationPairwiseTrainer

Losses
------

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationPairwiseLoss
   :members: compute

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.MSEDifferenceLoss
.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationKLLoss



Masked Language Model
*********************

Sampler
-------

.. autoxpmconfig:: xpmir.mlm.samplers.MLMSampler

Trainer
-------

.. autoxpmconfig:: xpmir.mlm.trainer.MLMTrainer

Losses
------

.. autoxpmconfig:: xpmir.mlm.trainer.MLMLoss
.. autoxpmconfig:: xpmir.mlm.trainer.CrossEntropyLoss
