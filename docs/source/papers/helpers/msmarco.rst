MS-Marco helpers
================

.. currentmodule:: xpmir.papers.helpers.msmarco

Base
----

The base experiment is controlled by the following configuration

.. autoclass:: ValidationSample
   :members:


Re-ranking
----------

.. autoclass:: RerankerMSMarcoV1Configuration
   :members:


.. autofunction:: v1_docpairs_sampler
.. autofunction:: v1_validation_dataset
.. autofunction:: v1_tests
.. autofunction:: v1_passages
.. autofunction:: v1_devsmall
.. autofunction:: v1_dev
.. autofunction:: v1_measures

First stage rankers
-------------------

(in progress)
