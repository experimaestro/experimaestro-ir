import torch
from experimaestro import Config
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path
import os
import json
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    List,
    NamedTuple,
    Optional,
    Protocol,
    TypeVar,
    Type,
    TYPE_CHECKING,
)
from shutil import rmtree
from xpmir.utils import easylog
from xpmir.letor.metrics import Metric, Metrics
from experimaestro.utils import cleanupdir
from contextlib import contextmanager

if TYPE_CHECKING:
    from xpmir.rankers import LearnableScorer
    from xpmir.letor.optim import ScheduledOptimizer
    from xpmir.letor.trainers import Trainer

logger = easylog()


class Loss(NamedTuple):
    """A loss"""

    name: str
    value: torch.Tensor
    weight: float


class TrainState:
    """Represents a training state for serialization"""

    epoch: int
    """The epoch"""

    steps: int
    """The number of steps (each epoch is composed of sptes)"""

    def __init__(
        self,
        model: "LearnableScorer",
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

        torch.save(self.model.state_dict(), path / "model.pth")
        torch.save(self.trainer.state_dict(), path / "trainer.pth")
        torch.save(self.optimizer.state_dict(), path / "optimizer.pth")

        self.path = path

    def load(self, path, onlyinfo=False):
        """Loads the state from disk"""
        if not onlyinfo:
            self.model.load_state_dict(torch.load(path / "model.pth"))
            self.trainer.load_state_dict(torch.load(path / "trainer.pth"))
            self.optimizer.load_state_dict(torch.load(path / "optimizer.pth"))

        with (path / "info.json").open("rt") as fp:
            self.load_state_dict(json.load(fp))
        self.path = path
        self.cached = True

    def copy_model(self, path: Path):
        assert self.path is not None
        for name in ["model.pth", "info.json"]:
            os.link(self.path / name, path / name)


HookType = TypeVar("HookType")


class TrainingHook(Config):
    """Base class for all training hooks"""

    pass


class StepTrainingHook(TrainingHook):
    """Base class for hooks called at each epoch (before/after)"""

    def after(self, state: "TrainerContext"):
        pass

    def before(self, state: "TrainerContext"):
        pass


class LearnContext:
    """The global learning context"""


class TrainerContext:
    """Contains all the information about the training context
    for a spefic

    This object is used when training to provide scorers and losses'
    with extra information - as well as the possibility to add
    regularization losses
    """

    metrics: Optional[Metrics]
    """Metrics to be reported"""

    _losses: Optional[List[Loss]]
    """Regularization losses to be added to the main loss"""

    hooksmap: Dict[Type, List[TrainingHook]]
    """Map of hooks"""

    _scope: List[str]
    """Scope for metric names"""

    PREFIX = "epoch-"

    def __init__(
        self,
        logpath: Path,
        path: Path,
        max_epoch: int,
        steps_per_epoch: int,
        trainer,
        ranker: "LearnableScorer",
        optimizer: "ScheduledOptimizer",
    ):
        self.path = path
        self.logpath = logpath
        self.max_epoch = max_epoch
        self.steps_per_epoch = steps_per_epoch
        self._writer = None
        self._scope = []
        self.hooksmap = DefaultDict(lambda: [])
        self._losses = None

        self.state = TrainState(ranker, trainer, optimizer)

    def add_hook(self, hook):
        for cls in hook.__class__.__mro__:
            self.hooksmap[cls].append(hook)

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
            rmtree(self.oldstate.path)
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

    def hooks(self, cls: Type[HookType]) -> List[HookType]:
        """Returns all the hooks"""
        return self.hooksmap.get(cls, [])  # type: ignore

    def call_hooks(self, cls: Type, method: Callable, *args, **kwargs):
        for hook in self.hooks(cls):
            method(hook, *args, **kwargs)

    def add_loss(self, loss: Loss):
        assert self._losses is not None, "This call should be in the losses context"
        self._losses.append(loss)

    @contextmanager
    def losses(self):
        try:
            self._losses = []
            yield self._losses
        finally:
            self._losses = None

    @contextmanager
    def step(self):
        try:
            self.state.optimizer.zero_grad()
            self.metrics = Metrics()
            yield self.metrics
            self.state.optimizer.optimizer_step(self)
            self.state.optimizer.scheduler_step(self)
        finally:
            self.metrics = None

    def add_metric(self, metric: Metric):
        assert self.metrics is not None, "Not within an optimization step"
        self.metrics.add(metric)

    @contextmanager
    def scope(self, name: str):
        try:
            self._scope.append(name)
            yield
        finally:
            self._scope.pop()
