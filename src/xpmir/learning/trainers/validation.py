import torch
import json
from typing import Dict, Any
from pathlib import Path

from experimaestro import Annotated, Param, pathgenerator

from xpmir.learning import SampleIterator
from xpmir.learning.batchers import Batcher
from xpmir.learning.learner import (
    Learner,
    LearnerListener,
    LearnerListenerStatus,
    TrainerContext,
    TrainState,
)
from xpmir.learning.optim import ModuleLoader
from xpmir.letor.records import BaseRecords
from xpmir.letor.trainers import LossTrainer
from xpmir.learning.metrics import Metrics, ScalarMetric
from xpmir.utils.logging import easylog

logger = easylog()


class TrainerValidationLoss(LearnerListener):
    """Generic trainer-based loss validation"""

    data: Param[SampleIterator]
    """The dataset to use"""

    batcher: Param[Batcher] = Batcher()
    """How to batch samples together"""

    batch_size: Param[int]
    """Batch size"""

    trainer: Param[LossTrainer]
    """The trainer"""

    warmup: Param[int] = -1
    """How many epochs before actually computing the validation loss"""

    bestpath: Annotated[Path, pathgenerator("best")]
    """Path to the best checkpoints"""

    info: Annotated[Path, pathgenerator("info.json")]
    """Path to the JSON file that contains the metric values at each epoch"""

    validation_interval: Param[int] = 1
    """Epochs between each validation"""

    early_stop: Param[int] = 0
    """Number of epochs without improvement after which we stop learning.
    Should be a multiple of validation_interval or 0 (no early stopping)"""

    def __validate__(self):
        assert (
            self.early_stop % self.validation_interval == 0
        ), "Early stop should be a multiple of the validation interval"

    def initialize(self, learner: Learner, context: TrainerContext):
        super().initialize(learner, context)
        self.scope = f"validation/{self.id}"
        self.bestpath.mkdir(exist_ok=True, parents=True)
        self.batcher_worker = self.batcher.initialize(self.batch_size)

        # Checkpoint start
        try:
            with self.info.open("rt") as fp:
                self.top: Dict[str, Any] = json.load(fp)
        except Exception:
            self.top = None

    def init_task(self, learner: "Learner", dep):
        return dep(
            ModuleLoader(
                value=learner.model,
                path=self.bestpath / TrainState.MODEL_PATH,
            )
        )

    def update_metrics(self, metrics: Dict[str, float]):
        if self.top:
            # Just use another key
            metrics[f"{self.id}/final"] = self.top["value"]

    def should_stop(self, epoch=0):
        if self.early_stop > 0 and self.top:
            epochs_since_imp = (epoch or self.context.epoch) - self.top["epoch"]
            if epochs_since_imp >= self.early_stop:
                return LearnerListenerStatus.STOP

        return LearnerListenerStatus.DONT_STOP

    def reducer(self, records: BaseRecords, metrics: Metrics):
        """Combines a forward and backard

        This method can be implemented by specific trainers that use the gradient.
        In that case the regularizer losses should be taken into account with
        `self.add_losses`.
        """
        # Restrict losses to this context

        with self.context.losses() as losses:
            # Compute the loss(es)
            self.trainer.train_batch(records)

            # Aggregate with previous values
            nrecords = len(records)
            total_loss = 0.0
            names = []

            for loss in losses:
                total_loss += loss.weight * loss.value
                names.append(loss.name)
                metrics.add(
                    ScalarMetric(
                        f"loss/{loss.name}", float(loss.value.item()), nrecords
                    )
                )

            # Reports the main metric
            if len(names) > 1:
                names.sort()
                metrics.add(ScalarMetric("loss", float(total_loss.item()), nrecords))

            return metrics

    def __call__(self, state: TrainState):
        # Check that we did not stop earlier (when loading from checkpoint / if other
        # listeners have not stopped yet)
        if self.should_stop(state.epoch - 1) == LearnerListenerStatus.STOP:
            return LearnerListenerStatus.STOP

        if state.epoch % self.validation_interval == 0:
            # Compute validation metrics
            metrics = Metrics()
            state.model.eval()

            with torch.no_grad():
                for batch in self.data.__batch_iter__(self.batch_size):
                    self.batcher_worker.reduce(
                        batch, self.reducer, metrics, raise_oom=False
                    )

            metrics.report(state.step, self.context.writer, self.id)

            # Get the current value
            if len(metrics.metrics) == 1:
                value = next(iter(metrics.metrics.values())).compute()
            else:
                value = metrics.metrics["loss"].compute()

            # Update the top validation
            if state.epoch >= self.warmup:
                topstate = self.top
                if topstate is None or value > topstate["value"]:
                    # Save the new top JSON
                    self.top = {"value": value, "epoch": self.context.epoch}

                    # Copy in corresponding directory
                    logger.info(f"Saving the checkpoint {state.epoch}")
                    self.context.copy(self.bestpath)

            # Update information
            with self.info.open("wt") as fp:
                json.dump(self.top, fp)

        # Early stopping?
        return self.should_stop()
