Papers
======

To ease the reproduction of papers, and the upload of full models on HuggingFace,
the `xpmir.papers` package can be used.


Configuration
-------------

Papers experimental parameters are defined by data classes. The main
one is :py:class:`xpmir.papers.helpers.PaperExperiment` that
defines an id (for experimaestro), a title and a description.
These informations can be used e.g. when uploading the trained
models on HuggingFace.

.. autoclass:: xpmir.papers.helpers.PaperExperiment
    :members:

.. autoclass:: xpmir.papers.helpers.NeuralIRExperiment
    :members:

Helpers
-------

Pipelines factorize the code necessary to run some experiments:
for instance, training re-rankers on MS-Marco is usually performed
with similar training data, evaluation datasets.


.. toctree::
    :maxdepth: 2

    helpers/index
    helpers/msmarco



Implemented papers
------------------

This page give some information about reproduction of papers based on XPMIR.

- `xpmir/cross-encoders <https://github.com/xpmir/cross-encoders>`_:

    - monoBERT

.. toctree::
   :maxdepth: 2

   monobert
   splade
