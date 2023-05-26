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

.. automodule:: xpmir.learning.schedulers
    :members:
