Running IR experiments
----------------------

The module `experimaestro.experiments` contain code factorizing boilerplate for
launching experiments, which is specialized in `xpmir` with specific experiment
helpers.

For instance, one can define a standard IR experiments that learns (with tensorboard),
evaluates a model on a different metrics and upload it to HuggingFace.

The experiment can be started with

.. code-block: sh

    experimaestro run-experiment --run-mode normal full.yaml


IR experiment
=============

.. automodule:: xpmir.experiments.ir
    :members:
