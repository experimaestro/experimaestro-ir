
Distillation
************


Sampler
-------

.. autoxpmconfig:: xpmir.letor.distillation.samplers.DistillationPairwiseSampler
.. autoxpmconfig:: xpmir.letor.distillation.samplers.PairwiseHydrator
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

Samplers
--------

.. autoclass:: xpmir.letor.distillation.samplers.PairwiseDistillationSample
    :members: documents, query

.. autoxpmconfig:: xpmir.letor.distillation.samplers.PairwiseDistillationSamples
.. autoxpmconfig:: xpmir.letor.distillation.samplers.PairwiseDistillationSamplesTSV
