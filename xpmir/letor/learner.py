import logging
import tempfile
import json
import os
from pathlib import Path
from typing import Dict, List
from datamaestro_text.data.ir import Adhoc
from experimaestro import task, Config, param, Param, pathoption, pathgenerator
from experimaestro.annotations import option
from experimaestro.notifications import tqdm
from experimaestro.utils import cleanupdir
from typing_extensions import Annotated
from xpmir.utils import EasyLogger
from xpmir.evaluation import evaluate
from xpmir.letor import Random
from xpmir.letor.trainers import TrainContext, TrainState, Trainer
from xpmir.rankers import LearnableScorer, Retriever, ScoredDocument, Scorer


class LearnerListener(Config):
    """Hook for learner

    Performs some operations after a learning epoch"""

    def initialize(self, key: str, learner: "Learner", context: TrainContext):
        self.key = key
        self.learner = learner
        self.context = context

    def __call__(self, state) -> bool:
        """Process and returns whether the training process should stop"""
        return False

    def update_metrics(self, metrics: Dict[str, float]):
        """Add metrics"""
        pass


class SavedScorer(Scorer):
    path: Param[Path]
    config: Param[Config]

    # When using the best validation model
    _model = None

    @property
    def model(self):
        if self._model is None:
            state = TrainState()
            state.load(self.path)
            self._model = state.ranker

        return self._model

    def rsv(self, query: str, documents: List[ScoredDocument]) -> List[ScoredDocument]:
        return self.model.rsv(query, documents)


class ValidationListener(LearnerListener):
    """Learning validation early-stopping

    Computes a validation metric and stores the best result

    Attributes:
        warmup: Number of warmup epochs
        early_stop: Maximum number of epochs without improvement on validation
        validation: How to compute the validation metric
        validation_interval: interval for computing validation metrics
        metrics: Dictionary whose keys are the metrics to record, and boolean
            values whether the best performance checkpoint should be kept for
            the associated metric
    """

    metrics: Param[Dict[str, bool]] = {"map": True}
    dataset: Param[Adhoc]
    retriever: Param[Retriever]
    validation_interval: Param[int] = 1
    warmup: Param[int] = -1
    bestpath: Annotated[Path, pathgenerator("best")]
    info: Annotated[Path, pathgenerator("info.json")]
    early_stop: Param[int] = 20

    def initialize(self, key: str, learner: "Learner", context: TrainContext):
        super().initialize(key, learner, context)

        self.retriever.initialize()
        self.bestpath.mkdir(exist_ok=True, parents=True)

        # Checkpoint start
        try:
            with self.info.open("rt") as fp:
                self.top = json.load(fp)  # type: Dict[str, Dict[str, float]]
        except Exception:
            self.top = {}

    def update_metrics(self, metrics: Dict[str, float]):
        if self.top:
            # Just use another key
            for metric in self.metrics.keys():
                metrics[f"{metric}/{self.key}"] = self.top[self.key]

    def getscorer(self, key: str) -> Scorer:
        """Return a scorer corresponding to the best (validation-wise) one for the given metric"""
        assert self.metrics.get(
            key, False
        ), f"Metric {key} is not part of recorded metrics"
        return SavedScorer(path=self.bestpath / key, config=self)

    def __call__(self, state):
        if state.epoch % self.validation_interval == 0:
            # Compute validation metrics
            means, _ = evaluate(
                None, self.retriever, self.dataset, list(self.metrics.keys())
            )

            for metric, keep in self.metrics.items():
                value = means[metric]

                self.context.writer.add_scalar(
                    f"{self.key}/{metric}", value, state.epoch
                )

                # Update the top validation
                if keep and state.epoch >= self.warmup:
                    topstate = self.top.get(metric, None)
                    if topstate is None or value > topstate["value"]:
                        # Save the new top JSON
                        self.top[metric] = {"value": value, "epoch": self.context.epoch}
                        with self.info.open("wt") as fp:
                            json.dump(self.top, fp)

                        # Copy in corresponding directory
                        self.context.copy(self.bestpath / metric)

        # Early stopping?
        if self.early_stop > 0 and self.top:
            epochs_since_imp = self.context.epoch - max(
                info["epoch"] for info in self.top.values()
            )
            if epochs_since_imp >= self.early_stop:
                return False

        # No, proceed...
        return True


# Checkpoints
@task()
class Learner(EasyLogger):
    """Learns a model

    The learner task is generic, and takes two main arguments:
    (1) the scorer defines the model (e.g. DRMM), and
    (2) the trainer defines the loss (e.g. pointwise, pairwise, etc.)

    Attributes:

        max_epoch: Maximum training epoch
        early_stop: Maximum number of epochs without improvement on validation
        checkpoint_interval: Number of epochs between each checkpoint
        scorer: The scorer to learn
        trainer: The trainer used to learn the parameters of the scorer
        listeners: learning process listeners (e.g. validation or other metrics)
        random: Random generator
    """

    # Training
    random: Param[Random]

    max_epoch: Param[int] = 1000
    trainer: Param[Trainer]
    scorer: Param[LearnableScorer]

    listeners: Param[Dict[str, LearnerListener]]

    # Checkpoints
    checkpoint_interval: Param[int] = 1

    # Paths
    logpath: Annotated[Path, pathgenerator("runs")]
    checkpointspath: Annotated[Path, pathgenerator("checkpoints")]

    # The Trainer
    def execute(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)

        self.only_cached = False

        # Initialize the scorer and trainer
        self.logger.info("Scorer initialization")
        self.scorer.initialize(self.random.state)

        # Initialize the listeners
        context = TrainContext(self.logpath, self.checkpointspath)
        for key, listener in self.listeners.items():
            listener.initialize(key, self, context)

        self.logger.info("Trainer initialization")
        self.trainer.initialize(self.random.state, self.scorer, context)

        self.logger.info("Starting to train")

        current = 0
        state = None
        with tqdm(
            self.trainer.iter_train(self.max_epoch), total=self.max_epoch
        ) as states:
            for state in states:
                # Report progress
                states.update(state.epoch - current)

                if state.epoch >= 0 and not self.only_cached:
                    message = f"epoch {state.epoch}"
                    if state.cached:
                        self.logger.debug(f"[train] [cached] {message}")
                    else:
                        self.logger.debug(f"[train] {message}")

                if state.epoch == -1:
                    continue

                if not state.cached:
                    # Save checkpoint if needed
                    if state.epoch % self.checkpoint_interval == 0:
                        context.save_checkpoint()

                    # Call listeners
                    stop = False
                    for listener in self.listeners.values():
                        stop = listener(state) and stop

                    if stop:
                        self.logger.warn(
                            "stopping after epoch {epoch} ({early_stop} epochs since "
                            "all listeners asked for it"
                        )

                # Stop if max epoch is reached
                if context.epoch >= self.max_epoch:
                    self.logger.warn(
                        "stopping after epoch {max_epoch} (max_epoch)".format(
                            **self.__dict__
                        )
                    )
                    break

            # End of the learning process
            if state is not None and not state.cached:
                # Set the hyper-parameters
                metrics = {}
                for listener in self.listeners.values():
                    listener.update_metrics(metrics)
                context.writer.add_hparams(self.__tags__, metrics)
