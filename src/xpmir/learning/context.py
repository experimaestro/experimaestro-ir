import torch
from experimaestro import Config
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
import os
import json
from typing import (
    List,
    NamedTuple,
    Optional,
    TYPE_CHECKING,
)
from shutil import rmtree
from xpmir.context import Context, InitializationHook
from xpmir.utils.utils import easylog
from xpmir.learning.devices import DeviceInformation
from xpmir.learning.metrics import Metric, Metrics
from experimaestro.utils import cleanupdir
from contextlib import contextmanager

if TYPE_CHECKING:
    from xpmir.learning.optim import ScheduledOptimizer, Module
    from xpmir.learning.trainers import Trainer

logger = easylog()


class Loss(NamedTuple):
    """A loss"""

    name: str
    value: torch.Tensor
    weight: float


class TrainState:
    """Represents a training state for serialization"""

    MODEL_PATH = "model.pth"

    epoch: int
    """The epoch"""

    steps: int
    """The number of steps (each epoch is composed of sptes)"""

    def __init__(
        self,
        model: "Module",
        trainer: "Trainer",
        optimizer: "ScheduledOptimizer",
        epoch=0,
        steps=0,
    ):
        # Initialize the state
        self.model = model
        self.trainer = trainer
        self.optimizer = optimizer

        self.epoch = epoch
        self.steps = steps

        # Was it loaded from disk?
        self.cached = False

        # Was it saved?
        self.path = None

    def copy(self):
        return TrainState(self.model, self.trainer, self.optimizer, **self.state_dict())

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "steps": self.steps,
        }

    @property
    def step(self):
        """Returns the step for logging (number of steps)"""
        return self.steps

    def load_state_dict(self, state):
        self.epoch = state.get("epoch", 0)
        self.steps = state.get("steps", 0)

    def save(self, path):
        """Save the state"""
        cleanupdir(path)

        with (path / "info.json").open("wt") as fp:
            json.dump(self.state_dict(), fp)

        torch.save(self.model.state_dict(), path / TrainState.MODEL_PATH)
        torch.save(self.trainer.state_dict(), path / "trainer.pth")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pth")

        self.path = path

    def load(self, path, onlyinfo=False):
        """Loads the state from disk"""
        if not onlyinfo:
            self.model.load_state_dict(torch.load(path / TrainState.MODEL_PATH))
            self.trainer.load_state_dict(torch.load(path / "trainer.pth"))
            self.optimizer.load_state_dict(torch.load(path / "optimizer.pth"))

        with (path / "info.json").open("rt") as fp:
            self.load_state_dict(json.load(fp))
        self.path = path
        self.cached = True

    def copy_model(self, path: Path):
        assert self.path is not None
        for name in [TrainState.MODEL_PATH, "info.json"]:
            os.link(self.path / name, path / name)


class TrainingHook(Config):
    """Base class for all training hooks"""

    pass


class StepTrainingHook(TrainingHook):
    """Base class for hooks called at each step (before/after)"""

    def after(self, state: "TrainerContext"):
        """Called after a training step"""

    def before(self, state: "TrainerContext"):
        """Called before a training step"""


class InitializationTrainingHook(InitializationHook):
    """Base class for hooks called at each epoch (before/after)"""

    def after(self, state: "TrainerContext"):
        pass

    def before(self, state: "TrainerContext"):
        pass


class TrainerContext(Context):
    """Contains all the information about the training context
    for a spefic

    This object is used when training to provide models and losses'
    with extra information - as well as the possibility to add
    regularization losses
    """

    metrics: Optional[Metrics]
    """Metrics to be reported"""

    _losses: Optional[List[Loss]]
    """Regularization losses to be added to the main loss"""

    _scope: List[str]
    """Scope for metric names"""

    PREFIX = "epoch-"

    def __init__(
        self,
        device_information: DeviceInformation,
        logpath: Path,
        path: Path,
        max_epoch: int,
        steps_per_epoch: int,
        trainer,
        model: "Module",
        optimizer: "ScheduledOptimizer",
    ):
        super().__init__(device_information)
        self.path = path
        self.logpath = logpath
        self.max_epoch = max_epoch
        self.steps_per_epoch = steps_per_epoch
        self._writer = None
        self._scope = []
        self._losses = None

        self.state = TrainState(model, trainer, optimizer)

    @property
    def writer(self):
        """Returns a tensorboard writer

        by default, purges the entries beside the current epoch
        """
        if self._writer is None:
            self._writer = SummaryWriter(self.logpath, purge_step=self.state.step)
        return self._writer

    @property
    def epoch(self):
        return self.state.epoch

    @property
    def steps(self):
        return self.state.steps

    def nextepoch(self):
        self.oldstate = self.state
        self.state = self.oldstate.copy()
        self.state.epoch += 1

    def nextbatch(self):
        self.state.steps += 1

    def load_bestcheckpoint(self, max_epoch: int):
        """Try to find the best checkpoint to load (the highest lower than
        the epoch target)"""
        # Find all the potential epochs
        epochs = []
        for f in self.path.glob(f"{TrainerContext.PREFIX}*"):
            epoch = int(f.name[len(TrainerContext.PREFIX) :])
            if epoch <= max_epoch:
                epochs.append(epoch)
        epochs.sort(reverse=True)

        # Try to load the first one
        for epoch in epochs:
            logger.info("Loading from checkpoint at epoch %d", epoch)
            path = self.path / f"{TrainerContext.PREFIX}{epoch:08d}"

            try:
                self.state.load(path)
                return True
            except NotImplementedError:
                logger.error("Not removing checkpoint")
                raise
            except Exception:
                rmtree(path)
                logger.exception("Cannot load from epoch %d", epoch)

        return False

    def save_checkpoint(self):
        # Serialize
        path = self.path / f"{TrainerContext.PREFIX}{self.epoch:08d}"
        if self.state.path is not None:
            # No need to save twice
            return

        # Save
        self.state.save(path)

        # Cleanup if needed
        if self.oldstate and self.oldstate.path:
            try:
                rmtree(self.oldstate.path)
            except OSError as e:
                # We continue the learning process in those cases
                logger.error("OS Error while trying to remove directory %s", e)

            self.oldstate = None

    def copy(self, path: Path):
        """Copy the state into another folder"""
        if self.state.path is None:
            self.save_checkpoint()

        trainpath = self.state.path
        assert trainpath is not None

        if path:
            cleanupdir(path)
            self.state.copy_model(path)

    def add_loss(self, loss: Loss):
        assert (
            self._losses is not None
        ), "This should be called in the context where loss is computed"
        self._losses.append(loss)

    @contextmanager
    def losses(self):
        previous = self._losses
        try:
            self._losses = []
            yield self._losses
        finally:
            self._losses = previous

    @contextmanager
    def step(self, metrics):
        try:
            self.state.optimizer.zero_grad()
            self.metrics = Metrics()
            yield self.metrics
            self.state.optimizer.optimizer_step(self)
            self.state.optimizer.scheduler_step(self)
            metrics.merge(self.metrics)
        finally:
            self.metrics = None

    def add_metric(self, metric: Metric):
        assert self.metrics is not None, "Not within an optimization step"
        if self._scope:
            metric.key = "/".join(s for s in self._scope if s) + "/" + metric.key
        self.metrics.add(metric)

    @contextmanager
    def scope(self, name: str):
        try:
            self._scope.append(name)
            yield
        finally:
            self._scope.pop()
