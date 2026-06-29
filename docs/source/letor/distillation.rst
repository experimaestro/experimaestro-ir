Knowledge distillation
======================

Knowledge distillation trains a student model to mimic the output distribution
of a stronger teacher model. This is commonly used to transfer the accuracy of
a cross-encoder teacher into a faster bi-encoder or sparse student.

.. contents:: On this page
   :local:
   :depth: 2


Samplers
--------

Samplers that pair documents with teacher scores for distillation training.

.. autoxpmconfig:: xpmir.letor.distillation.samplers.DistillationPairwiseSampler
.. autoxpmconfig:: xpmir.letor.samplers.TeacherModelBasedHardNegativesTripletSampler

Trainer
-------

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationPairwiseTrainer

Pairwise losses
---------------

Losses operating on pairs of documents with teacher scores.

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationPairwiseLoss
   :members: compute

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.MSEDifferenceLoss
.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationKLLoss
.. autoxpmconfig:: xpmir.letor.distillation.listwise.ListwiseInfoNCE

Data types
----------

.. autoclass:: xpmir.letor.distillation.samplers.PairwiseDistillationSample
    :members: documents, query
