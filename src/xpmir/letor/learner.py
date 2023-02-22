import logging
import torch
import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple
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
from xpmir.context import Hook, InitializationHook
from xpmir.letor.batchers import RecoverableOOMError
from xpmir.utils.utils import EasyLogger, easylog, foreach
from xpmir.evaluation import evaluate
from xpmir.letor import DEFAULT_DEVICE, Device, DeviceInformation, Random
from xpmir.letor.trainers import Trainer
from xpmir.letor.context import (
    StepTrainingHook,
    TrainState,
    TrainerContext,
)
from xpmir.letor.metrics import Metrics
from xpmir.rankers import (
    AbstractLearnableScorer,
    Retriever,
)
from xpmir.letor.optim import ParameterOptimizer, ScheduledOptimizer

logger = easylog()


class LearnerListener(Config):
    """Hook for learner

    Performs some operations after a learning epoch"""

    def initialize(self, key: str, learner: "Learner", context: TrainerContext):
        self.key = key
        self.learner = learner
        self.context = context

    def __call__(self, state: TrainerContext) -> bool:
        """Process and returns whether the training process should stop

        Returns:
            bool: True if the learning process should stop
        """
        return False

    def update_metrics(self, metrics: Dict[str, float]):
        """Add metrics"""
        pass

    def taskoutputs(self, learner: "Learner"):
        """Outputs from this listeners"""
        return None


class ValidationListener(LearnerListener):
    """Learning validation early-stopping

    Computes a validation metric and stores the best result. If early_stop is
    set (> 0), then it signals to the learner that the learning process can
    stop.
    """

    metrics: Param[Dict[str, bool]] = {"map": True}
    """Dictionary whose keys are the metrics to record, and boolean
            values whether the best performance checkpoint should be kept for
            the associated metric ([parseable by ir-measures](https://ir-measur.es/))"""

    dataset: Param[Adhoc]
    """The dataset to use"""

    retriever: Param[Retriever]
    """The retriever for validation"""

    warmup: Param[int] = -1
    """How many epochs before actually computing the metric"""

    bestpath: Annotated[Path, pathgenerator("best")]
    """Path to the best checkpoints"""

    last_checkpoint_path: Annotated[Path, pathgenerator("last_checkpoint")]
    """Path to the last checkpoints"""

    store_last_checkpoint: Param[bool] = False
    """Besides the model with the best performance, whether store the last
    checkpoint of the model or not"""

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

    def initialize(self, key: str, learner: "Learner", context: TrainerContext):
        super().initialize(key, learner, context)

        self.retriever.initialize()
        self.bestpath.mkdir(exist_ok=True, parents=True)
        if self.store_last_checkpoint:
            self.last_checkpoint_path.mkdir(exist_ok=True, parents=True)

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

    def monitored(self) -> Iterator[str]:
        return [key for key, monitored in self.metrics.items() if monitored]

    def taskoutputs(self, learner: "Learner"):
        """Experimaestro outputs: returns the best checkpoints for each
        metric"""
        res = {
            key: copyconfig(learner.scorer, checkpoint=str(self.bestpath / key))
            for key, store in self.metrics.items()
            if store
        }
        if self.store_last_checkpoint:
            res["last_checkpoint"] = copyconfig(
                learner.scorer, checkpoint=str(self.last_checkpoint_path)
            )

        return res

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
                    np.array(list(details[metric].values()), dtype=np.float32),
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
                                f"Saving the checkpoint {state.epoch}"
                                f" for metric {metric}"
                            )
                            self.context.copy(self.bestpath / metric)

            if self.store_last_checkpoint:
                logging.info(f"Saving the last checkpoint {state.epoch}")
                self.context.copy(self.last_checkpoint_path)

            # Update information
            with self.info.open("wt") as fp:
                json.dump(self.top, fp)

        # Early stopping?
        return self.should_stop()


class LearnerOutput(NamedTuple):
    """The data structure for the output of a learner. It contains a dictionary
    where the key is the name of the listener and the value is the output of
    that listener"""

    listeners: Dict[str, Any]


# Checkpoints
class Learner(Task, EasyLogger):
    """Model Learner

    The learner task is generic, and takes two main arguments: (1) the scorer
    defines the model (e.g. DRMM), and (2) the trainer defines how the model
    should be trained (e.g. pointwise, pairwise, etc.)

    When submitted, it returns a dictionary based on the `listeners`
    """

    # Training
    random: Param[Random]
    """The random generator"""

    trainer: Param[Trainer]
    """Specifies how to train the model"""

    scorer: Param[AbstractLearnableScorer]
    """Defines the model that scores documents"""

    max_epochs: Param[int] = 1000
    """Maximum number of epochs"""

    steps_per_epoch: Param[int] = 128
    """Number of steps for one epoch (after each epoch results are logged)"""

    use_fp16: Param[bool] = False
    """Use mixed precision when training"""

    optimizers: Param[List[ParameterOptimizer]]
    """The list of parameter optimizers"""

    listeners: Param[Dict[str, LearnerListener]]
    """Listeners are in charge of handling the validation of the model, and
    saving the relevant checkpoints"""

    checkpoint_interval: Param[int] = 1
    """Number of epochs between each checkpoint"""

    logpath: Annotated[Path, pathgenerator("runs")]
    """The path to the tensorboard logs"""

    checkpointspath: Annotated[Path, pathgenerator("checkpoints")]
    """The path to the checkpoints"""

    device: Meta[Device] = DEFAULT_DEVICE
    """The device(s) to be used for the model"""

    hooks: Param[List[Hook]] = []
    """Global learning hooks


    :class:`Initialization hooks <xpmir.context.InitializationHook>` are called
    before and after the initialization of the trainer and listeners.
    """

    def __validate__(self):
        assert self.optimizers, "At least one optimizer should be defined"
        return super().__validate__()

    def taskoutputs(self) -> LearnerOutput:
        """Object returned when submitting the task"""
        return LearnerOutput(
            listeners={
                key: listener.taskoutputs(self)
                for key, listener in self.listeners.items()
            }
        )

    def execute(self):
        self.device.execute(self.device_execute)

    def device_execute(self, device_information: DeviceInformation):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        for handler in logger.handlers:
            handler.setLevel(logging.INFO)

        self.optimizer = ScheduledOptimizer()
        self.only_cached = False
        self.context = TrainerContext(
            device_information,
            self.logpath,
            self.checkpointspath,
            self.max_epochs,
            self.steps_per_epoch,
            self.trainer,
            self.scorer,
            self.optimizer,
        )

        for hook in self.hooks:
            self.context.add_hook(hook)

        # Call hooks
        foreach(
            self.context.hooks(InitializationHook),
            lambda hook: hook.before(self.context),
        )

        # Sets the random seed
        seed = self.random.state.randint((2**32) - 1)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Initialize the scorer and trainer
        self.logger.info("Scorer initialization")
        self.scorer.initialize(self.random.state)

        # Initialize the context and the listeners
        self.trainer.initialize(self.random.state, self.context)
        for key, listener in self.listeners.items():
            listener.initialize(key, self, self.context)

        self.logger.info("Moving to device %s", device_information.device)
        self.scorer.to(device_information.device)
        self.trainer.to(device_information.device)
        num_training_steps = self.max_epochs * self.steps_per_epoch
        self.optimizer.initialize(
            self.optimizers, num_training_steps, self.scorer, self.use_fp16
        )

        foreach(
            self.context.hooks(InitializationHook),
            lambda hook: hook.after(self.context),
        )

        self.logger.info("Starting to train")

        current = 0
        state = None

        with tqdm(
            total=self.max_epochs, desc=f"Training ({self.max_epochs} epochs)"
        ) as tqdm_epochs:
            for state in self.iter_train(device_information):
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

    def iter_train(self, device_information) -> Iterator[TrainState]:
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
                total=self.steps_per_epoch,
                ncols=100,
                desc=f"Train - epoch #{self.context.epoch}",
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
                            with self.context.step(metrics):
                                # Call the hook epoch hook
                                foreach(
                                    self.context.hooks(StepTrainingHook),
                                    lambda hook: hook.before(self.context),
                                )

                                # Computes the gradient
                                with torch.autocast(
                                    device_information.device.type,
                                    enabled=self.use_fp16,
                                ):
                                    self.trainer.process_batch(batch)

                                # Update metrics and counter
                                pbar.update(1)
                                break
                        except RecoverableOOMError:
                            logger.warning(
                                "Recoverable OOM detected"
                                " - re-running the training step"
                            )
                            continue

                    foreach(
                        self.context.hooks(StepTrainingHook),
                        lambda hook: hook.after(self.context),
                    )

                # Yields the current state (after one epoch)
                yield self.context.state

                # Report metrics over the epoch
                metrics.report(
                    self.context.state.step,
                    self.context.writer,
                    "train",
                )
