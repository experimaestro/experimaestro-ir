
Hooks
=====


Inference
---------

.. currentmodule:: xpmir.context

.. autoxpmconfig:: Hook

.. autoxpmconfig:: InitializationHook
    :members: before, after

Learning
--------

.. currentmodule:: xpmir.learning.context

Hooks can be used to modify the learning process

.. autoxpmconfig:: TrainingHook
.. autoxpmconfig:: InitializationTrainingHook
    :members: before, after

.. autoxpmconfig:: StepTrainingHook
    :members: before, after



Distributed
-----------

Hooks can be used to distribute a model over GPUs

.. autoxpmconfig:: xpmir.distributed.DistributableModel
    :members: distribute_models

.. autoxpmconfig:: xpmir.distributed.DistributedHook
