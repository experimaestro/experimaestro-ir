
Distillation
************


Sampler
-------

.. autoxpmconfig:: xpmir.letor.distillation.samplers.DistillationPairwiseSampler
.. autoxpmconfig:: xpmir.letor.samplers.TeacherModelBasedHardNegativesTripletSampler

Trainer
-------

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationPairwiseTrainer

Losses
------

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationPairwiseLoss
   :members: compute

.. autoxpmconfig:: xpmir.letor.distillation.pairwise.MSEDifferenceLoss
.. autoxpmconfig:: xpmir.letor.distillation.pairwise.DistillationKLLoss
.. autoxpmconfig:: xpmir.letor.distillation.listwise.ListwiseInfoNCE

Samplers
--------

.. autoclass:: xpmir.letor.distillation.samplers.PairwiseDistillationSample
    :members: documents, query
