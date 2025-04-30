Learning
--------
.. _Learning:

Learning is handled by various classes :

- The `Learner`_ is the main class that runs the full process
- `Listeners`_ are used for validation or other monitoring tasks
- `Trainers`_ that iterate over batches of data
- :ref:`Optimization <Optimization>` deals with parameters (selecting, gradient descent, etc.)


.. toctree::
   :hidden:

   optimization


Learner
=======

The main class is the Learner task; when submitted to the scheduler,
returns a :py:class:`LearnerOutput <xpmir.learning.learner.LearnerOutput>`.

.. autoxpmconfig:: xpmir.learning.learner.Learner
.. autonamedtuple:: xpmir.learning.learner.LearnerOutput

Trainers
========

Trainers are responsible for defining the the way to train
a learnable scorer.

.. autoxpmconfig:: xpmir.learning.trainers.Trainer
.. autoxpmconfig:: xpmir.learning.trainers.multiple.MultipleTrainer

.. autoxpmconfig:: xpmir.letor.trainers.LossTrainer
   :members: process_microbatch

Listeners
=========

.. _Listeners:

Listeners can be used to monitor the learning process

.. autoxpmconfig:: xpmir.learning.learner.LearnerListener
   :members: __call__

.. autoxpmconfig:: xpmir.learning.context.ValidationHook
.. autoxpmconfig:: xpmir.learning.trainers.validation.TrainerValidationLoss