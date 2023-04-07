Papers
======


.. highlight:: bash

experimaestro-IR contains some reproduction of existing papers, which are listed
here:

- **monobert**: Passage Re-ranking with BERT, (Rodrigo Nogueira, Kyunghyun Cho). 2019
  https://arxiv.org/abs/1901.04085


To get the list of papers

.. code-block:: bash

    $ xpmir papers --help

To get the list of experiments for a paper (here, monobert)

.. code-block:: bash

    $ xpmir papers monobert --help

Runs an experiment (monobert with "small" configuration, training on msmarco)

.. code-block:: bash

    $ xpmir papers monobert msmarco --configuration small /path/to/workdir

with some overriding arguments

.. code-block:: bash

    $ xpmir papers monobert msmarco --configuration small /path/to/workdir learner.lr=2.0e-6 learner.num_warmup_steps=20

You can look at arguments by using the `--show` option

.. code-block:: bash

    $ xpmir papers monobert msmarco --configuration small /path/to/workdir --show


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

.. toctree::
   :maxdepth: 2

   monobert
   splade
