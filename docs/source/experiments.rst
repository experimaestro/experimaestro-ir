Running IR experiments
----------------------

The module `experimaestro.experiments` contain code factorizing boilerplate for
launching experiments, which is specialized in `xpmir` with specific experiment
helpers.

For instance, one can define a standard IR experiments that learns (with tensorboard),
evaluates a model on a different metrics and upload it to HuggingFace.

With `full.yaml` located in the same folder as `experiment.py`

.. code-block:: yaml

    file: experiment
    learning_rate: 1e-4

The experiment can be started with

.. code-block:: sh

    experimaestro run-experiment --run-mode normal full.yaml

Learning experiment
===================

Generic learning experiments can be conducted with the
:py:module:`xpmir.experiments.learning` module that allows to easily use a
Tensorboard service.


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
=============

Example
*******

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
***

.. automodule:: xpmir.experiments.ir
    :members:
