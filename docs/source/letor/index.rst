Learning to rank
----------------

.. toctree::
   :maxdepth: 2

   samplers
   trainers
   optimization


Learning to rank is handled by various classes :

- the learner is the main class that runs the full process
- learner listeners are used for validation
- trainers that iterate over batches of data

The main class is the Learner task.

.. autoxpmconfig:: xpmir.learning.learner.Learner
.. autonamedtuple:: xpmir.learning.learner.LearnerOutput


Listeners
=========

Listeners can be used to monitor the learning process

.. autoxpmconfig:: xpmir.learning.learner.LearnerListener
   :members: __call__

.. autoxpmconfig:: xpmir.letor.learner.ValidationListener

Scorers
=======

Scorers are able to give a score to a (query, document) pair. Among the
scorers, some are have learnable parameters.

.. autoxpmconfig:: xpmir.rankers.Scorer
   :members: initialize, rsv, to, eval, getRetriever
.. autoxpmconfig:: xpmir.rankers.RandomScorer
.. autoxpmconfig:: xpmir.rankers.AbstractModuleScorer
.. autoxpmconfig:: xpmir.rankers.LearnableScorer

.. autofunction:: xpmir.rankers.scorer_retriever

Retrievers
==========

Scores can be used as retrievers through a :py:class:`xpmir.rankers.TwoStageRetriever`
