Learning to rank
----------------


Scorers
=======

Scorers are able to give a score to a (query, document) pair. Among the
scorers, some are have learnable parameters.

.. autoxpmconfig:: xpmir.rankers.Scorer
.. autoxpmconfig:: xpmir.rankers.RandomScorer
.. autoxpmconfig:: xpmir.rankers.LearnableScorer

Retrievers
==========

Scores can be used as retrievers through

.. autoxpmconfig:: xpmir.rankers.TwoStageRetriever


Trainers
========

Trainers are responsible for defining the loss (given a learnable scorer)

.. automodule:: xpmir.letor.trainers.Trainer

Sampler
=======

How to sample learning batches.

.. autosummary:: xpmir.letor.samplers.Sampler
