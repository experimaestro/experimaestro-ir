Samplers
--------

Samplers provide samples in the form of *records*. They all inherit from:

.. autoxpmconfig:: xpmir.letor.samplers.Sampler
.. autoclass:: xpmir.letor.samplers.SerializableIterator


Pointwise
=========

.. autoxpmconfig:: xpmir.letor.samplers.PointwiseSampler
    :members: pointwise_iter

.. autoxpmconfig:: xpmir.letor.samplers.PointwiseModelBasedSampler

Pairwise
=========

.. autoxpmconfig:: xpmir.letor.samplers.PairwiseSampler
.. autoxpmconfig:: xpmir.letor.samplers.PairwiseModelBasedSampler
.. autoxpmconfig:: xpmir.documents.samplers.BatchwiseRandomSpanSampler

.. autoxpmconfig:: xpmir.letor.samplers.TripletBasedSampler
.. autoxpmconfig:: xpmir.letor.samplers.PairwiseDatasetTripletBasedSampler


Records for training
====================

.. automodule:: xpmir.letor.records
    :members: PointwiseRecord, PairwiseRecord
