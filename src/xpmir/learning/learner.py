from enum import Enum
import logging
import torch
from pathlib import Path
from typing import Dict, Iterator, List, NamedTuple, Any
from experimaestro import (
    Task,
    Config,
    Param,
    pathgenerator,
    Annotated,
    tqdm,
    Meta,
)
import numpy as np
from xpmir.context import Hook, InitializationHook
from xpmir.utils.utils import EasyLogger, easylog, foreach
from xpmir.learning.devices import DEFAULT_DEVICE, Device, DeviceInformation
from xpmir.learning import Random
from xpmir.learning.trainers import Trainer
from xpmir.learning.context import (
    StepTrainingHook,
    TrainState,
    TrainerContext,
)
from xpmir.learning.metrics import Metrics

from .optim import Module, ModuleLoader, ParameterOptimizer, ScheduledOptimizer
from .batchers import RecoverableOOMError

logger = easylog()


class LearnerListenerStatus(Enum):
    NO_DECISION = 0
    STOP = 1
    DONT_STOP = 2

    def update(self, other: "LearnerListenerStatus") -> "LearnerListenerStatus":
        return LearnerListenerStatus(max(self.value, other.value))


class LearnerListener(Config):
    """Hook for learner

    Performs some operations after a learning epoch"""

    id: Meta[str]
    """Unique ID to identify the listener (ignored for signature)"""

    def initialize(self, learner: "Learner", context: TrainerContext):
        self.learner = learner
        self.context = context

    def __call__(self, state: TrainState) -> LearnerListenerStatus:
        """Process and returns whether the training process should stop"""
        return LearnerListenerStatus.NO_DECISION

    def update_metrics(self, metrics: Dict[str, float]):
        """Add metrics"""
        pass

    def task_outputs(self, learner: "Learner", dep):
        """Outputs from this listeners

        :param learner: The learner object
        :param dep: The function that adds a dependency
        """
        return None

    def init_task(self, learner: "Learner", dep):
        """Returns the initialization task that loads the associated checkpoint

        :param learner: The learner object
        :param dep: The function that adds a dependency
        """
        return None


class LearnerOutput(NamedTuple):
    """The data structure for the output of a learner. It contains a dictionary
    where the key is the name of the listener and the value is the output of
    that listener"""

    listeners: Dict[str, Any]

    learned_model: Module


class Learner(Task, EasyLogger):
    """Model Learner

    The learner task is generic, and takes two main arguments: (1) the model
    defines the model (e.g. DRMM), and (2) the trainer defines how the model
    should be trained (e.g. pointwise, pairwise, etc.)

    When submitted, it returns a dictionary based on the `listeners`
    """

    # Training
    random: Param[Random]
    """The random generator"""

    trainer: Param[Trainer]
    """Specifies how to train the model"""

    model: Param[Module]
    """Defines the model that scores documents"""

    max_epochs: Param[int] = 1000
    """Maximum number of epochs"""

    steps_per_epoch: Param[int] = 128
    """Number of steps for one epoch (after each epoch results are logged)"""

    use_fp16: Param[bool] = False
    """Use mixed precision when training"""

    optimizers: Param[List[ParameterOptimizer]]
    """The list of parameter optimizers"""

    listeners: Param[List[LearnerListener]]
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

    use_pretasks: Meta[bool] = False
    """Use deprected pre-tasks as the output"""

    def __validate__(self):
        assert self.optimizers, "At least one optimizer should be defined"
        assert len(set(listener.id for listener in self.listeners)) == len(
            self.listeners
        ), "IDs of listeners should be unique"
        return super().__validate__()

    def task_outputs(self, dep) -> LearnerOutput:
        """Object returned when submitting the task"""
        if self.use_pretasks:
            logging.warn("Using deprecated pre-tasks in Learner")
            return LearnerOutput(
                listeners={
                    listener.id: listener.task_outputs(self, dep)
                    for listener in self.listeners
                },
                learned_model=ModuleLoader.construct(
                    self.model, self.last_checkpoint_path / TrainState.MODEL_PATH, dep
                ),
            )

        return LearnerOutput(
            listeners={
                listener.id: listener.init_task(self, dep)
                for listener in self.listeners
            },
            learned_model=dep(
                ModuleLoader(
                    value=self.model,
                    path=self.last_checkpoint_path / TrainState.MODEL_PATH,
                )
            ),
        )

    @property
    def last_checkpoint_path(self):
        return self.checkpointspath / "last"

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
            self.model,
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
        self.model.initialize()

        # Initialize the context and the listeners
        self.trainer.initialize(self.random.state, self.context)
        for listener in self.listeners:
            listener.initialize(self, self.context)

        self.logger.info("Moving to device %s", device_information.device)
        self.model.to(device_information.device)
        self.trainer.to(device_information.device)
        num_training_steps = self.max_epochs * self.steps_per_epoch
        self.optimizer.initialize(
            self.optimizers, num_training_steps, self.model, self.use_fp16
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
                    self.context.copy(self.last_checkpoint_path)

                # Call listeners
                decision = LearnerListenerStatus.NO_DECISION
                for listener in self.listeners:
                    # listener.__call__ returns True if we should stop
                    decision = decision.update(listener(state))

                if decision == LearnerListenerStatus.STOP:
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
                for listener in self.listeners:
                    listener.update_metrics(metrics)
                self.context.writer.add_hparams(getattr(self, "__tags__", {}), metrics)

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
