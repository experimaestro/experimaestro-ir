Optimization
============


Optimizers
----------


.. autoxpmconfig:: xpmir.learning.optim.Optimizer
.. autoxpmconfig:: xpmir.learning.optim.Adam
.. autoxpmconfig:: xpmir.learning.optim.AdamW

.. autoxpmconfig:: xpmir.learning.optim.ParameterOptimizer
.. autoxpmconfig:: xpmir.learning.optim.ParameterFilter
.. autoxpmconfig:: xpmir.learning.optim.Module




Batching
--------

.. autoxpmconfig:: xpmir.learning.batchers.Batcher
.. autoxpmconfig:: xpmir.learning.batchers.PowerAdaptativeBatcher

Devices
-------

The devices configuration allow to select both the device to use for computation and
the way to use it (i.e. multi-gpu settings).

.. autoxpmconfig:: xpmir.letor.devices.Device

.. autoxpmconfig:: xpmir.letor.devices.CudaDevice


Schedulers
----------

.. automodule:: xpmir.learning.schedulers
    :members:
