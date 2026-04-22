Running experiments
===================

The ``xpmir.experiments`` module provides decorators and helper classes that
reduce the boilerplate needed to define a reproducible IR experiment. An
experiment is a single Python function decorated with
:func:`~xpmir.experiments.ir.ir_experiment` (or ``learning_experiment`` for
generic learning tasks), paired with a YAML configuration file.

Given a YAML file ``full.yaml``:

.. code-block:: yaml

    file: experiment
    learning_rate: 1e-4

The experiment can be started with:

.. code-block:: sh

    experimaestro run-experiment --run-mode normal full.yaml


Learning experiment
-------------------

Generic learning experiments (not IR-specific). Provides a
:class:`~xpmir.experiments.learning.LearningExperimentHelper` with
Tensorboard logging.

.. code-block:: python

    from experimaestro.experiments import configuration
    from xpmir.experiments.learning import PaperResults, learning_experiment, LearningExperimentHelper

    @configuration
    class Configuration:
        #: Default learning rate
        learning_rate: float = 1e-3

    @learning_experiment()
    def run(
        helper: LearningExperimentHelper, cfg: Configuration
    ) -> PaperResults:
        ...


IR experiment
-------------

IR-specific experiments that add evaluation and model-upload capabilities on
top of the learning experiment.

Example
^^^^^^^

.. code-block:: python

    from experimaestro.experiments import configuration
    from xpmir.experiments.ir import PaperResults, ir_experiment, IRExperimentHelper

    @configuration
    class Configuration:
        #: Default learning rate
        learning_rate: float = 1e-3

    @ir_experiment()
    def run(
        helper: IRExperimentHelper, cfg: Configuration
    ) -> PaperResults:
        ...

        return PaperResults(
            models={"my-model@RR10": outputs.listeners[validation.id]["RR@10"]},
            evaluations=tests,
            tb_logs={"my-model@RR10": learner.logpath},
        )

API
^^^

.. automodule:: xpmir.experiments.ir
    :members:
