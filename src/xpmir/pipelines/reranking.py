from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Protocol, List
from experimaestro import tagspath, experiment
from experimaestro.launchers import Launcher
from datamaestro_text.data.ir import Adhoc, AdhocDocuments
from xpmir.letor.devices import Device
from xpmir.letor import Random
from xpmir.rankers import LearnableScorer, Retriever
from xpmir.evaluation import EvaluationsCollection
from xpmir.letor.trainers import Trainer
from xpmir.letor.optim import get_optimizers, Optimizers
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.context import Hook


class RetrieverFactory(Protocol):
    def __call__(self, scorer: LearnableScorer, documents: AdhocDocuments) -> Retriever:
        ...


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

    retriever_factory: RetrieverFactory
    """The first-stage retriever used for testing"""

    steps_per_epoch: int
    """Number of learning steps per epoch"""

    max_epochs: int
    """Number of epochs"""

    validation_dataset: Adhoc
    """The validation dataset"""

    validation_metrics: Dict[str, bool]
    """Metrics used for validation. If the corresponding value is False, the
    metric value is computed but no validation is performed on it"""

    # Test part
    tests: EvaluationsCollection
    """Final evaluation for each model"""

    # Optional parameters
    launcher: Optional[Launcher] = None
    """Launcher for learning"""

    evaluate_launcher: Optional[Launcher] = None
    """Launcher for evaluating the models"""

    device: Optional[Device] = None
    """Device for the computation"""

    random: Optional[Random] = None
    """Random generator"""

    validation_interval: Optional[int] = 1
    """Epochs between each validation"""

    validation_retriever_factory: Optional[RetrieverFactory] = None

    runs_path: Optional[Path] = None

    hooks: Optional[List[Hook]] = field(default_factory=lambda: [])

    def run(self, scorer: LearnableScorer):
        """Train a the reranking part of a two-stage ranker


        Parameters
        ----------

        scorer: a scorer with a given trainer, before evaluating
        """

        # Sets default values if needed
        val_retriever_factory = (
            self.validation_retriever_factory or self.retriever_factory
        )

        random = self.random or Random()

        # The validation listener will evaluate the full retriever
        # (1st stage + reranker) and keep the best performing model
        # on the validation set
        validation = ValidationListener(
            dataset=self.validation_dataset,
            retriever=val_retriever_factory(scorer, self.validation_dataset.documents),
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
            # The hook used for evaluation
            hooks=self.hooks,
        )

        # Submit job and link
        runs_path = self.runs_path or (experiment.current().resultspath / "runs")
        outputs = learner.submit(launcher=self.launcher)
        (runs_path / tagspath(learner)).symlink_to(learner.logpath)

        # Evaluate the neural model
        for metric_name, monitored in self.validation_metrics.items():
            if monitored:
                best = outputs.listeners["bestval"][metric_name]
                self.tests.evaluate_retriever(
                    lambda documents: self.retriever_factory(best, documents),
                    self.evaluate_launcher,
                )

        # return the output of the learner in order to get the information about
        # the best retriever
        return outputs
