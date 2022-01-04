Learning to rank
----------------

.. toctree::
   :maxdepth: 2

   samplers
   optimization

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

Trainers are responsible for defining the the way to train
a learnable scorer.

.. autoxpmconfig:: xpmir.letor.trainers.Trainer
.. autoxpmconfig:: xpmir.letor.trainers.pointwise.PointwiseTrainer
.. autoxpmconfig:: xpmir.letor.trainers.pairwise.PairwiseTrainer
