import logging
import torch
import json
from pathlib import Path
from typing import Dict, Iterable, List
from datamaestro_text.data.ir import Adhoc
from experimaestro import (
    Task,
    Config,
    Param,
    Meta,
    pathgenerator,
)
import numpy as np
from experimaestro.notifications import tqdm
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

    def taskoutputs(self, learner: "Learner"):
        """Outputs from this listeners"""
        return None


class SavedScorer(Scorer):
    """Generic scorer that has been saved"""

    ranker: Param[LearnableScorer]
    checkpoint: Meta[Path]

    def __postinit__(self):
        state = TrainState()
        state.ranker = self.ranker
        self.ranker.initialize(np.random.RandomState())
        state.optimizer = None
        state.load(Path(self.checkpoint))
        return state.ranker

    def to(self, device):
        self.ranker.to(device)

    def rsv(
        self, query: str, documents: Iterable[ScoredDocument], keepcontent=False
    ) -> List[ScoredDocument]:
        return self.ranker.rsv(query, documents)


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
                metrics[f"{self.key}/final/{metric}"] = self.top[metric]["value"]

    def taskoutputs(self, learner: "Learner"):
        """Experimaestro outputs"""
        return {
            key: SavedScorer(ranker=learner.scorer, checkpoint=str(self.bestpath / key))
            for key, store in self.metrics.items()
            if store
        }

    def __call__(self, state):
        if state.epoch % self.validation_interval == 0:
            # Compute validation metrics
            means = evaluate(self.retriever, self.dataset, list(self.metrics.keys()))

            for metric, keep in self.metrics.items():
                value = means[metric]

                self.context.writer.add_scalar(
                    f"{self.key}/{metric}", value, state.epoch
                )

                # Update the top validation
                if state.epoch >= self.warmup:
                    topstate = self.top.get(metric, None)
                    if topstate is None or value > topstate["value"]:
                        # Save the new top JSON
                        self.top[metric] = {"value": value, "epoch": self.context.epoch}

                        # Copy in corresponding directory
                        if keep:
                            logging.info(
                                f"Saving the checkpoint {state.epoch} for metric {metric}"
                            )
                            self.context.copy(self.bestpath / metric)

            # Update information
            with self.info.open("wt") as fp:
                json.dump(self.top, fp)

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
class Learner(Task, EasyLogger):
    """Model Learner

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

    def taskoutputs(self):
        return {
            "listeners": {
                key: listener.taskoutputs(self)
                for key, listener in self.listeners.items()
            }
        }

    # The Trainer
    def execute(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.only_cached = False

        # Initialize the scorer and trainer
        self.logger.info("Scorer initialization")
        self.scorer.initialize(self.random.state)

        # Initialize the listeners
        context = TrainContext(self.logpath, self.checkpointspath)
        for key, listener in self.listeners.items():
            listener.initialize(key, self, context)

        self.logger.info("Trainer initialization")
        seed = self.random.state.randint((2 ** 32) - 1)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Initialize the trainer
        self.trainer.initialize(self.random.state, self.scorer, context)

        self.logger.info("Starting to train")

        current = 0
        state = None

        with tqdm(total=self.max_epoch) as tqdm_epochs:
            for state in self.trainer.iter_train(self.max_epoch):
                # Report progress
                tqdm_epochs.update(state.epoch - current)
                current = state.epoch

                if state.epoch >= 0 and not self.only_cached:
                    message = f"epoch {state.epoch}"
                    if state.cached:
                        self.logger.debug(f"[train] [cached] {message}")
                    else:
                        self.logger.debug(f"[train] {message}")

                if state.epoch == -1:
                    continue

                if not state.cached and state.epoch % self.checkpoint_interval == 0:
                    # Save checkpoint if needed
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
