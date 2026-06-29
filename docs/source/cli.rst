Command-line interface
======================

XPMIR extends the ``experimaestro`` CLI with IR-specific commands.

Running experiments
-------------------

Experiments are run using the ``run-experiment`` command (see
:doc:`experiments` for details on how to define experiments):

.. code-block:: sh

    experimaestro run-experiment --run-mode normal experiment.yaml

HuggingFace utilities
---------------------

Pre-load a HuggingFace model into the local cache (useful before running on a
cluster without internet access):

.. code-block:: sh

    xpmir huggingface preload HF_MODEL_ID
