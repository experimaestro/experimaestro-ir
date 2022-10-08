from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from experimaestro import tagspath, experiment
from experimaestro.launchers import Launcher
from datamaestro_text.data.ir import Adhoc
from xpmir.letor.devices import Device
from xpmir.letor import Random
from xpmir.rankers import LearnableScorer, Retriever
from xpmir.evaluation import Evaluate, Evaluations, EvaluationsCollection
from xpmir.letor.trainers import Trainer
from xpmir.letor.optim import get_optimizers, Optimizers
from xpmir.letor.learner import Learner, ValidationListener


@dataclass()
class RerankingPipeline:
    """Holds the information necessary to run a re-ranking pipeline.

    The :py:meth:`run` method can be called to run the full pipeline for a given
    (learnable) scorer.
    """

    # Training part
    trainer: Trainer
    """The way to train the learnable scorer"""

    optimizers: Optimizers
    """How to optimize during the training process"""

    base_retriever: Retriever
    """The first-stage retriever used for testing"""

    steps_per_epoch: int
    """Number of learning steps per epoch"""

    max_epochs: int
    """Number of epochs"""

    validation_dataset: Adhoc
    """The validation dataset"""

    validation_metrics: Dict[str, bool]
    """Metrics used for validation. If the corresponding value is False, the metric value is computed but no validation is performed on it"""

    validation_batch_size: int
    """Batch size when rescoring documents during the validation"""

    # Test part
    tests: EvaluationsCollection
    """Final evaluation for each model"""

    # Optional parameters
    test_batch_size: Optional[int] = None
    launcher: Optional[Launcher] = None
    device: Optional[Device] = None
    random: Optional[Random] = None
    validation_interval = 1
    base_retriever_val: Optional[Retriever] = None
    runs_path: Optional[Path] = None

    def run(self, scorer: LearnableScorer):
        """Train a the reranking part of a two-stage ranker


        Parameters
        ----------

        scorer: a scorer with a given trainer, before evaluating
        """

        # Sets default values if needed
        base_retriever_val = self.base_retriever_val or self.base_retriever
        runs_path = self.runs_path or (experiment.current().resultspath / "runs")

        test_batch_size = self.test_batch_size or self.validation_batch_size
        random = self.random or Random()

        # The validation listener will evaluate the full retriever
        # (1st stage + reranker) and keep the best performing model
        # on the validation set
        validation = ValidationListener(
            dataset=self.validation_dataset,
            retriever=base_retriever_val.getReranker(
                scorer, self.validation_batch_size
            ),
            validation_interval=self.validation_interval,
            metrics=self.validation_metrics,
        )

        # The learner defines all what is needed
        # to perform several gradient steps
        learner = Learner(
            # Misc settings
            device=self.device,
            random=random,
            # How to train the model
            trainer=self.trainer,
            # The model to train
            scorer=scorer,
            # Optimization settings
            steps_per_epoch=self.steps_per_epoch,
            optimizers=get_optimizers(self.optimizers),
            max_epochs=self.max_epochs,
            # The listeners (here, for validation)
            listeners={"bestval": validation},
        )
        outputs = learner.submit(launcher=self.launcher)
        (runs_path / tagspath(learner)).symlink_to(learner.logpath)

        # Evaluate the neural model
        for metric_name, monitored in self.validation_metrics.items():
            if monitored:
                best = outputs.listeners["bestval"][metric_name]
                retriever = self.base_retriever.getReranker(
                    best, test_batch_size, device=self.device
                )
                self.tests.evaluate_retriever(retriever)
