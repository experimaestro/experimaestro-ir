
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

.. currentmodule:: xpmir.letor.context

Hooks can be used to modify the learning process

.. autoxpmconfig:: TrainingHook
.. autoxpmconfig:: InitializationTrainingHook
    :members: before, after

.. autoxpmconfig:: StepTrainingHook
    :members: before, after
