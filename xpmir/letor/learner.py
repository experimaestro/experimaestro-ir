import logging
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, List
from datamaestro_text.data.ir import Adhoc
from experimaestro import task, config, param, progress, pathoption
from experimaestro.annotations import option
from experimaestro.utils import cleanupdir
from xpmir.evaluation import evaluate
from xpmir.letor import Random
from xpmir.letor.trainers import TrainContext, TrainState, Trainer
from xpmir.rankers import LearnableScorer, Retriever, ScoredDocument, Scorer


class ValidationState(TrainState):
    # All metrics
    metrics: Dict[str, float]

    # Validation value
    value: float

    def __init__(self, state: TrainState = None):
        super().__init__(state)
        self.metrics = None
        self.value = None

    def __getstate__(self):
        dict = super().__getstate__()
        dict.update({"value": self.value, "metrics": self.metrics})
        return dict


class ValidationContext(TrainContext):
    STATETYPE = ValidationState

    def copy(self, path: Path):
        """Copy the state into another folder"""
        if self.state.path is None:
            self.save_checkpoint()

        trainpath = self.state.path

        if path:
            cleanupdir(path)
            for f in trainpath.rglob("*"):
                relpath = f.relative_to(trainpath)
                if f.is_dir():
                    (path / relpath).mkdir(exist_ok=True)
                else:
                    os.link(f, path / relpath)


@param("metric", default="map")
@param("dataset", type=Adhoc)
@param("retriever", type=Retriever)
@config()
class Validation:
    def initialize(self):
        self.retriever.initialize()
        self.metrics = [self.metric]

    def compute(self, state: ValidationState):
        # Evaluate
        mean, _ = evaluate(None, self.retriever, self.dataset, self.metrics)

        state.value = mean[self.metric]
        state.metrics = mean


# Training
@param("max_epoch", default=1000, help="Maximum training epoch")
@param(
    "early_stop",
    default=20,
    help="Maximum number of epochs without improvement on validation",
)
@param("warmup", default=-1, help="Number of warmup epochs")
@param("random", type=Random, help="Random generator")
@param("trainer", type=Trainer, help="The trainer used to learn the parameters")
@param("scorer", type=LearnableScorer, help="The scorer to learn")
# Validation
@param("validation", type=Validation, help="How to compute the validation metric")
# Checkpoints
@option(
    "checkpoint_interval", default=1, help="Number of epochs between each checkpoint"
)
@pathoption("checkpointspath", "checkpoints")
@pathoption("bestpath", "best")
@pathoption("logpath", "runs")
@task()
class Learner(Scorer):
    trainer: Trainer
    validation: Validation

    """The learner task is generic, and takes two main arguments:
    (1) the scorer defines the model (e.g. DRMM), and
    (2) the trainer defines the loss (e.g. pointwise, pairwise, etc.)
    """

    def execute(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)

        self.only_cached = False
        self.bestpath.mkdir(exist_ok=True, parents=True)

        # Initialize the scorer and trainer
        self.logger.info("Scorer initialization")
        self.scorer.initialize(self.random.state)
        self.validation.initialize()

        self.logger.info("Trainer initialization")
        context = ValidationContext(self.logpath, self.checkpointspath)
        self.trainer.initialize(self.random.state, self.scorer, context)

        # Top validation context
        try:
            top = context.newstate()
            top.load(self.bestpath, onlyinfo=True)
        except Exception:
            top = None

        self.logger.info("Starting to train")
        for state in self.trainer.iter_train(self.max_epoch):
            # Report progress
            progress(state.epoch / self.max_epoch)

            if state.epoch >= 0 and not self.only_cached:
                message = f"epoch {state.epoch}"
                if state.cached:
                    self.logger.debug(f"[train] [cached] {message}")
                else:
                    self.logger.debug(f"[train] {message}")

            if state.epoch == -1 and not self.initial_eval:
                continue

            # Compute validation metrics
            if not state.cached:
                # Compute validation metrics
                self.validation.compute(state)
                for metric in self.validation.metrics:
                    context.writer.add_scalar(
                        f"val/{metric}", state.metrics[metric], state.epoch
                    )

                # Save checkpoint if needed
                if state.epoch % self.checkpoint_interval == 0:
                    context.save_checkpoint()

                # Update the top validation
                if state.epoch >= self.warmup:
                    if top is None or state.value > top.value:
                        top = state
                        context.copy(self.bestpath)

            # Early stopping
            if top is not None:
                epochs_since_imp = context.epoch - top.epoch
                if self.early_stop > 0 and epochs_since_imp >= self.early_stop:
                    self.logger.warn(
                        "stopping after epoch {epoch} ({early_stop} epochs with no "
                        "improvement to validation metric)".format(
                            **state.__dict__, **self.__dict__
                        )
                    )
                    break

            # Early stopping
            if context.epoch >= self.max_epoch:
                self.logger.warn(
                    "stopping after epoch {max_epoch} (max_epoch)".format(
                        **self.__dict__
                    )
                )
                break

        if not state.cached:
            # Set the hyper-parameters
            context.writer.add_hparams(self.__tags__, state.metrics)

        self.logger.info("top validation epoch={} {}".format(top.epoch, top.value))

    _bestmodel = None

    @property
    def bestmodel(self):
        if self._bestmodel is None:
            context = ValidationContext(self.logpath, self.checkpointspath)
            top = context.newstate()
            top.load(self.bestpath, onlyinfo=False)
            self._bestmodel = top.ranker

        return self._bestmodel

    def rsv(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
        return self.bestmodel.rsv(query, documents)
