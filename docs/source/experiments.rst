Running IR experiments
----------------------

The module `xpmir.experiments` contain code factorizing boilerplate for launching experiments

For instance, one can define a standard IR experiments that learns (with tensorboard),
evaluates a model on a different metrics and upload it to HuggingFace.


Example
=======

An `experiment.py` file:

.. code-block:: py3

    from xpmir.experiments.ir import PaperResults, ir_experiment, ExperimentHelper
    from xpmir.papers import configuration

    @configuration
    class Configuration:
        #: Default learning rate
        learning_rate: float = 1e-3

    @ir_experiment()
    def run(
        helper: ExperimentHelper, cfg: Configuration
    ) -> PaperResults:
        ...

        return PaperResults(
            models={"my-model@RR10": outputs.listeners[validation.id]["RR@10"]},
            evaluations=tests,
            tb_logs={"my-model@RR10": learner.logpath},
        )


With `full.yaml` located in the same folder as `experiment.py`

.. code-block:: yaml

    file: experiment
    learning_rate: 1e-4

The experiment can be started with

.. code-block: sh

    xpmir run-experiment --run-mode normal full.yaml

Common handling
===============

.. automodule:: xpmir.experiments.cli
    :members:

IR experiment
=============

.. automodule:: xpmir.experiments.ir
    :members:
