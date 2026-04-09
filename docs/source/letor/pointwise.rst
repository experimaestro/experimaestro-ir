Pointwise learning
==================

In pointwise learning, each training instance is a single (query, document)
pair labelled with a relevance score. The model is trained to predict this
score directly (e.g. via regression or classification).

Sampler
-------

.. currentmodule:: xpmir.letor.samplers

.. autoxpmconfig:: PointwiseModelBasedSampler
