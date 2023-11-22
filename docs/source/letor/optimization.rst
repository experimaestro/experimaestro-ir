Optimization
============


Modules
-------


.. autoxpmconfig:: xpmir.learning.optim.Module
    :members:


The module loader can be used to load a checkpoint

.. autoxpmconfig:: xpmir.learning.optim.ModuleLoader


Optimizers
----------


.. autoxpmconfig:: xpmir.learning.optim.Optimizer
.. autoxpmconfig:: xpmir.learning.optim.Adam
.. autoxpmconfig:: xpmir.learning.optim.AdamW

.. autoxpmconfig:: xpmir.learning.optim.ParameterOptimizer
.. autoxpmconfig:: xpmir.learning.optim.ParameterFilter
.. autoxpmconfig:: xpmir.learning.optim.RegexParameterFilter


Parameters
----------

During learning, some parameter-specific treatments can be applied (e.g. freezing).


Selecting
*********

The classes below allow to select a subset of parameters.

.. autoxpmconfig:: xpmir.learning.parameters.InverseParametersIterator
.. autoxpmconfig:: xpmir.learning.parameters.ParametersIterator
.. autoxpmconfig:: xpmir.learning.parameters.SubParametersIterator

Freezing
********

.. autoxpmconfig:: xpmir.learning.hooks.LayerFreezer

Loading
*******

.. autoxpmconfig:: xpmir.learning.parameters.NameMapper
.. autoxpmconfig:: xpmir.learning.parameters.PrefixRenamer
.. autoxpmconfig:: xpmir.learning.parameters.PartialModuleLoader
.. autoxpmconfig:: xpmir.learning.parameters.SubModuleLoader


Batching
--------

.. autoxpmconfig:: xpmir.learning.batchers.Batcher
.. autoxpmconfig:: xpmir.learning.batchers.PowerAdaptativeBatcher

Devices
-------

The devices configuration allow to select both the device to use for computation and
the way to use it (i.e. multi-gpu settings).

.. autoxpmconfig:: xpmir.learning.devices.Device

.. autoxpmconfig:: xpmir.learning.devices.CudaDevice


Schedulers
----------

.. autoxpmconfig:: xpmir.learning.schedulers.Scheduler
.. autoxpmconfig:: xpmir.learning.schedulers.CosineWithWarmup
.. autoxpmconfig:: xpmir.learning.schedulers.LinearWithWarmup

Base classes
------------

.. autoxpmconfig:: xpmir.learning.base.Random
.. autoxpmconfig:: xpmir.learning.base.Sampler
.. autoxpmconfig:: xpmir.learning.trainers.Trainer
