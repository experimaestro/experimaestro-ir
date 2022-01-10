import logging
import torch
import json
import math
from pathlib import Path
from typing import Dict, Iterator, List
from datamaestro_text.data.ir import Adhoc
from experimaestro import (
    Task,
    Config,
    Param,
    copyconfig,
    pathgenerator,
    Annotated,
    tqdm,
    Meta,
)
import numpy as np
from experimaestro.notifications import tqdm
from xpmir.letor.batchers import RecoverableOOMError
from xpmir.utils import EasyLogger, foreach
from xpmir.evaluation import evaluate
from xpmir.letor import DEFAULT_DEVICE, Device, Random
from xpmir.letor.trainers import Trainer
from xpmir.letor.context import StepTrainingHook, TrainState, TrainerContext
from xpmir.letor.metrics import Metrics
from xpmir.rankers import LearnableScorer, Retriever, ScoredDocument, Scorer
from xpmir.letor.optim import ParameterOptimizer, ScheduledOptimizer


class LearnerListener(Config):
    """Hook for learner

    Performs some operations after a learning epoch"""

    def initialize(self, key: str, learner: "Learner", context: TrainerContext):
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


class ValidationListener(LearnerListener):
    """Learning validation early-stopping

    Computes a validation metric and stores the best result

    Attributes:
        warmup: Number of warmup epochs
        validation: How to compute the validation metric
        metrics: Dictionary whose keys are the metrics to record, and boolean
            values whether the best performance checkpoint should be kept for
            the associated metric
    """

    metrics: Param[Dict[str, bool]] = {"map": True}
    dataset: Param[Adhoc]
    retriever: Param[Retriever]
    warmup: Param[int] = -1
    bestpath: Annotated[Path, pathgenerator("best")]
    info: Annotated[Path, pathgenerator("info.json")]

    validation_interval: Param[int] = 1
    """Epochs between each validation"""

    early_stop: Param[int] = 0
    """Number of epochs without improvement after which we stop learning.
    Should be a multiple of validation_interval or 0 (no early stopping)"""

    def __validate__(self):
        assert (
            self.early_stop % self.validation_interval == 0
        ), "Early stop should be a multiple of the validation interval"

    def initialize(self, key: str, learner: "Learner", context: TrainerContext):
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
        """Experimaestro outputs: returns the best checkpoints for each
        metric"""
        return {
            key: copyconfig(learner.scorer, checkpoint=str(self.bestpath / key))
            for key, store in self.metrics.items()
            if store
        }

    def should_stop(self, epoch=0):
        if self.early_stop > 0 and self.top:
            epochs_since_imp = (epoch or self.context.epoch) - max(
                info["epoch"] for key, info in self.top.items() if self.metrics[key]
            )
            return epochs_since_imp >= self.early_stop

        # No, proceed...
        return False

    def __call__(self, state: TrainerContext):
        # Check that we did not stop earlier (when loading from checkpoint / if other
        # listeners have not stopped yet)
        if self.should_stop(state.epoch - 1):
            return True

        if state.epoch % self.validation_interval == 0:
            # Compute validation metrics
            means, details = evaluate(
                self.retriever, self.dataset, list(self.metrics.keys()), True
            )

            for metric, keep in self.metrics.items():
                value = means[metric]

                self.context.writer.add_scalar(
                    f"{self.key}/{metric}/mean", value, state.step
                )

                self.context.writer.add_histogram(
                    f"{self.key}/{metric}",
                    np.array(list(details[metric].values())),
                    state.step,
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
        return self.should_stop()


# Checkpoints
class Learner(Task, EasyLogger):
    """Model Learner

    The learner task is generic, and takes two main arguments:
    (1) the scorer defines the model (e.g. DRMM), and
    (2) the trainer defines how the model should be trained (e.g. pointwise, pairwise, etc.)

    Attributes:

        max_epoch: Maximum training epoch
        checkpoint_interval: Number of epochs between each checkpoint
        scorer: The scorer to learn
        trainer: The trainer used to learn the parameters of the scorer
        listeners: learning process listeners (e.g. validation or other metrics)
        random: Random generator
    """

    # Training
    random: Param[Random]

    trainer: Param[Trainer]
    scorer: Param[LearnableScorer]

    max_epochs: Param[int] = 1000
    """Maximum number of epochs"""

    steps_per_epoch: Param[int] = 128
    """Number of steps for one epoch (after each epoch results are logged)"""

    use_fp16: Param[bool] = False
    """Use mixed precision when training"""

    optimizers: Param[List[ParameterOptimizer]]
    """The list of parameter optimizers"""

    # Listen to learner
    listeners: Param[Dict[str, LearnerListener]]

    # Checkpoints
    checkpoint_interval: Param[int] = 1

    # Paths
    logpath: Annotated[Path, pathgenerator("runs")]
    checkpointspath: Annotated[Path, pathgenerator("checkpoints")]

    device: Meta[Device] = DEFAULT_DEVICE
    """The device(s) to be used for the model"""

    def __validate__(self):
        assert self.optimizers, "At least one optimizer should be defined"
        return super().__validate__()

    def taskoutputs(self):
        """Object returned when submitting the task"""
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

        self.only_cached = False
        self._device = self.device(logger)

        # Sets the random seed
        seed = self.random.state.randint((2 ** 32) - 1)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Initialize the scorer and trainer
        self.logger.info("Scorer initialization")
        self.scorer.initialize(self.random.state)

        # Initialize the optimizer
        num_training_steps = self.max_epochs * self.steps_per_epoch
        self.optimizer = ScheduledOptimizer(
            self.optimizers, num_training_steps, self.scorer, self.use_fp16
        )

        # Initialize the context and the listeners
        self.context = TrainerContext(
            self.logpath,
            self.checkpointspath,
            self.max_epochs,
            self.steps_per_epoch,
            self.trainer,
            self.scorer,
            self.optimizer,
        )
        self.trainer.initialize(self.random.state, self.context)
        for key, listener in self.listeners.items():
            listener.initialize(key, self, self.context)

        self.logger.info("Moving to device %s", self._device)
        self.scorer.to(self._device)
        self.trainer.to(self._device)

        self.logger.info("Starting to train")

        current = 0
        state = None

        with tqdm(total=self.max_epochs) as tqdm_epochs:
            for state in self.iter_train():
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
                    self.context.save_checkpoint()

                # Call listeners
                stop = True
                for listener in self.listeners.values():
                    # listener.__call__ returns True if we should stop
                    stop = listener(state) and stop

                if stop:
                    self.logger.warn(
                        "stopping after epoch {epoch} ({early_stop} epochs) since "
                        "all listeners asked for it"
                    )
                    break

                # Stop if max epoch is reached
                if self.context.epoch >= self.max_epochs:
                    self.logger.warn(
                        "stopping after epoch {max_epochs} (max_epoch)".format(
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
                self.context.writer.add_hparams(self.__tags__, metrics)

    def iter_train(self) -> Iterator[TrainState]:
        """Train iteration"""
        # Try to load a checkpoint

        if self.context.load_bestcheckpoint(self.max_epochs):
            yield self.context.state

        # Get an iterator over batches
        iter = self.trainer.iter_batches()

        while True:
            # Step to the next epoch
            self.context.nextepoch()

            # Train for an epoch
            with tqdm(
                leave=False,
                total=self.steps_per_epoch * self.max_epochs,
                ncols=100,
                desc=f"Train for {self.context.epoch} epochs",
            ) as pbar:
                # Put the model into training mode (just in case)
                self.context.state.model.train()

                # Epoch: loop over batches
                metrics = Metrics()
                for b in range(self.steps_per_epoch):
                    # Get the next batch
                    batch = next(iter)
                    self.context.nextbatch()

                    while True:
                        try:
                            # Computes the gradient, takes a step and collect metrics
                            with self.context.step() as step_metrics:
                                # Call the hook epoch hook
                                foreach(
                                    self.context.hooks(StepTrainingHook),
                                    lambda hook: hook.before(self.context),
                                )

                                # Computes the gradient
                                with torch.autocast(
                                    self._device.type, enabled=self.use_fp16
                                ):
                                    self.trainer.process_batch(batch)

                                # Update metrics and counter
                                pbar.update(1)
                                metrics.merge(step_metrics)

                                # Report metrics over the epoch
                                metrics.report(
                                    self.context.state.step,
                                    self.context.writer,
                                    "train",
                                )

                                # Yields the current state (after one epoch)
                                foreach(
                                    self.context.hooks(StepTrainingHook),
                                    lambda hook: hook.after(self.context),
                                )
                                yield self.context.state

                        except RecoverableOOMError:
                            continue
