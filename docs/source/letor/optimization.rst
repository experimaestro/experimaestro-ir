Optimization
============


Optimizers
----------


.. autoxpmconfig:: xpmir.letor.optim.Optimizer
.. autoxpmconfig:: xpmir.letor.optim.Adam
.. autoxpmconfig:: xpmir.letor.optim.AdamW

.. autoxpmconfig:: xpmir.letor.optim.ParameterOptimizer
.. autoxpmconfig:: xpmir.letor.optim.ParameterFilter
.. autoxpmconfig:: xpmir.letor.optim.Module




Batching
--------

.. autoxpmconfig:: xpmir.letor.batchers.Batcher
.. autoxpmconfig:: xpmir.letor.batchers.PowerAdaptativeBatcher

Devices
-------

The devices configuration allow to select both the device to use for computation and
the way to use it (i.e. multi-gpu settings).

.. autoxpmconfig:: xpmir.letor.devices.Device

.. autoxpmconfig:: xpmir.letor.devices.CudaDevice


Schedulers
----------

.. automodule:: xpmir.letor.schedulers
    :members:
