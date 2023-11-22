Samplers
--------

.. currentmodule:: xpmir.letor.samplers

Samplers provide samples in the form of *records*. They all inherit from:

.. autoxpmconfig:: Sampler
.. autoclass:: SerializableIterator


.. autoxpmconfig:: ModelBasedSampler


Pointwise
=========

.. autoxpmconfig:: PointwiseSampler
    :members: pointwise_iter

.. autoxpmconfig:: PointwiseModelBasedSampler

Pairwise
=========

.. autoxpmconfig:: PairwiseSampler
.. autoxpmconfig:: BatchwiseSampler
.. autoxpmconfig:: PairwiseModelBasedSampler
.. autoxpmconfig:: xpmir.documents.samplers.BatchwiseRandomSpanSampler

.. autoxpmconfig:: TripletBasedSampler
.. autoxpmconfig:: PairwiseDatasetTripletBasedSampler
.. autoxpmconfig:: PairwiseInBatchNegativesSampler
.. autoxpmconfig:: PairwiseSampleDatasetFromTSV
.. autoxpmconfig:: PairwiseSamplerFromTSV

Hard Negatives Sampling (Tasks)
===============================

.. autoxpmconfig:: ModelBasedHardNegativeSampler
.. autoxpmconfig:: TeacherModelBasedHardNegativesTripletSampler

Distillation
============

.. autoclass:: xpmir.letor.distillation.samplers.PairwiseDistillationSample
    :members: documents, query

.. autoxpmconfig:: xpmir.letor.distillation.samplers.PairwiseDistillationSamples
.. autoxpmconfig:: xpmir.letor.distillation.samplers.PairwiseDistillationSamplesTSV

Records for training
====================

.. automodule:: xpmir.letor.records
    :members: PointwiseRecord, PairwiseRecord


Document samplers
=================

Useful for pre-training or when learning index parameters (e.g. for FAISS).

.. currentmodule:: xpmir.documents.samplers
.. autoxpmconfig:: DocumentSampler
.. autoxpmconfig:: HeadDocumentSampler
.. autoxpmconfig:: RandomDocumentSampler
