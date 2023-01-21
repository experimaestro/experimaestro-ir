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

    $ xpmir papers monobert

Runs an experiment (monobert with "small" configuration, training on msmarco)

.. code-block:: bash

    $ xpmir papers monobert /path/to/workdir msmarco small

with some overriding arguments

.. code-block:: bash

    $ xpmir papers monobert small learner.lr=2.0e-6 learner.num_warmup_steps=20

You can look at arguments by using the `--show` option

.. code-block:: bash

    $ xpmir papers monobert small --show
