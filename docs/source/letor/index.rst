Learning to rank
----------------

.. toctree::
   :maxdepth: 2

   samplers
   trainers
   optimization


Learning to rank is handled by various classes :

- the learner is the main class that runs
-

The main class is the Learner task.

.. autoxpmconfig:: xpmir.letor.learner.Learner


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
