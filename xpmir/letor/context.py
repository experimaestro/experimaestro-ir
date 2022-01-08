from types import MethodType
import torch
from experimaestro import Config
from pathlib import Path
import os
import json
from torch.utils.tensorboard.writer import SummaryWriter
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generic,
    List,
    NamedTuple,
    Protocol,
    TypeVar,
    Type,
)
from shutil import rmtree
from xpmir.utils import easylog
from experimaestro.utils import cleanupdir


logger = easylog()


class Metric:
    key: str
    count: int
    """Number of samples for this value"""

    def __init__(self, key, count):
        self.key = key
        self.count = count

    def merge(self, other: "Metric"):
        assert other.__class__ is self.__class__
        self._merge(other)

    def _merge(self, other: "Metric"):
        raise NotImplementedError(f"_merge in {self.__class__}")

    def report(self, epoch: int, writer: SummaryWriter, prefix: str):
        raise NotImplementedError(f"report in {self.__class__}")


class ScalarMetric(Metric):
    """Represents a scalar metric"""

    sum: float

    def __init__(self, key: str, value: float, count: int):
        super().__init__(key, count)
        self.sum = value * count

    def _merge(self, other: "ScalarMetric"):
        self.sum += other.sum
        self.count += other.count

    def report(self, step: int, writer: SummaryWriter, prefix: str):
        writer.add_scalar(
            f"{prefix}/{self.key}",
            self.sum / self.count,
            step,
        )


class Metrics:
    """Utility class to accumulate a set of metrics over batches
    of (potentially) different sizes"""

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}

    def add(self, metric: Metric):
        if metric.key in self.metrics:
            self.metrics[metric.key].merge(metric)
        else:
            self.metrics[metric.key] = metric

    def merge(self, other: "Metrics"):
        for value in other.metrics.values():
            self.add(value)

    def report(self, epoch, writer, prefix):
        for value in self.metrics.values():
            value.report(epoch, writer, prefix)


class Loss(NamedTuple):
    name: str
    value: torch.Tensor
    weight: float


class TrainState:
    """Represents a training state, i.e. everything which is
    linked to an epoch"""

    epoch: int
    """The epoch"""

    batches: int
    """The number of batches (each epoch is composed of batches)"""

    samplecount: int = 0
    """Number of samples used so far"""

    def __init__(self, state: "TrainState" = None):
        # Model and optimizer
        self.ranker = (
            state.ranker if state else None
        )  # type: Optional["LearnableScorer"]

        self.optimizer = state.optimizer if state else None
        self.scheduler = state.scheduler if state else None
        self.sampler = state.sampler if state else None

        # Copy the state
        self.load_state_dict(state.state_dict() if state else {})

        # Was it loaded from disk?
        self.cached = False

        # Was it saved?
        self.path = None

    @property
    def step(self):
        """Returns the step for logging"""
        return self.samplecount

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "batches": self.batches,
            "samplecount": self.samplecount,
        }

    def load_state_dict(self, state):
        self.epoch = state.get("epoch", 0)
        self.batches = state.get("batches", 0)
        self.samplecount = state.get("samplecount", 0)

    def samplecount_add(self, count: int):
        self.samplecount += count

    def save(self, path):
        """Save the state"""
        cleanupdir(path)

        with (path / "info.json").open("wt") as fp:
            json.dump(self.state_dict(), fp)

        torch.save(
            {
                "model_state_dict": self.ranker.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer
                else None,
                "sampler_state_dict": self.sampler.state_dict()
                if self.sampler
                else None,
            },
            path / "checkpoint.pth",
        )

        self.path = path

    def load(self, path, onlyinfo=False):
        """Loads the state from disk"""
        if not onlyinfo:
            checkpoint = torch.load(path / "checkpoint.pth")
            assert self.ranker is not None
            self.ranker.load_state_dict(checkpoint["model_state_dict"])

            if self.sampler is not None:
                self.sampler.load_state_dict(checkpoint["sampler_state_dict"])

            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        with (path / "info.json").open("rt") as fp:
            self.load_state_dict(json.load(fp))

        self.path = path
        self.cached = True


HookType = TypeVar("HookType")


class TrainingHook(Config):
    """Base class for all training hooks"""

    pass


class StepTrainingHook(TrainingHook):
    """Base class for hooks called at each epoch (before/after)"""

    def after(self, state: "TrainContext"):
        pass

    def before(self, state: "TrainContext"):
        pass


class TrainContext:
    """Contains all the information about the training context

    This object is used when training to provide scorers and losses'
    with extra information - as well as the possibility to add
    regularization losses
    """

    metrics: Metrics
    """Metrics to be reported"""

    losses: List[Loss]
    """Regularization losses to be added to the main loss"""

    hooksmap: Dict[Type, List[TrainingHook]]
    """Map of hooks"""

    PREFIX = "epoch-"

    STATETYPE = TrainState

    def __init__(self, logpath: Path, path: Path, max_epoch: int):
        self.path = path
        self.logpath = logpath
        self.max_epoch = max_epoch
        self._writer = None
        self.hooksmap = DefaultDict(lambda: [])

        self.state = self.newstate()

        self.reset(True)

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

    @classmethod
    def newstate(cls, state=None):
        return cls.STATETYPE(state)

    @property
    def epoch(self):
        return self.state.epoch

    @property
    def batches(self):
        return self.state.batches

    def nextepoch(self):
        self.oldstate = self.state
        self.state = self.newstate(self.oldstate)
        self.state.epoch += 1

    def nextbatch(self):
        self.state.batches += 1

    def load_bestcheckpoint(self, target: int, ranker, optimizer, sampler):
        """Try to find the best checkpoint to load (the highest lower than
        the epoch target)"""
        # Find all the potential epochs
        epochs = []
        for f in self.path.glob(f"{TrainContext.PREFIX}*"):
            epoch = int(f.name[len(TrainContext.PREFIX) :])
            if epoch <= target:
                epochs.append(epoch)
        epochs.sort(reverse=True)

        # Try to load the first one
        for epoch in epochs:
            logger.info("Loading from checkpoint at epoch %d", epoch)
            path = self.path / f"{TrainContext.PREFIX}{epoch:08d}"

            try:
                self.state = self.newstate()
                self.state.ranker = ranker
                self.state.sampler = sampler
                self.state.optimizer = optimizer
                self.state.load(path)
                return True
            except NotImplementedError:
                logger.error("Not removing checkpoint")
                raise
            except Exception:
                rmtree(path)
                logger.exception(f"Cannot load from epoch %d", epoch)

        return False

    def save_checkpoint(self):
        # Serialize
        path = self.path / f"{TrainContext.PREFIX}{self.epoch:08d}"
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

        if path:
            cleanupdir(path)
            for f in trainpath.rglob("*"):
                relpath = f.relative_to(trainpath)
                if f.is_dir():
                    (path / relpath).mkdir(exist_ok=True)
                else:
                    os.link(f, path / relpath)

    def reset(self, epoch=False):
        if epoch:
            self.metrics = Metrics()
        self.losses = []

    def add_loss(self, loss: Loss):
        self.losses.append(loss)

    def hooks(self, cls: Type[HookType]) -> List[HookType]:
        """Returns all the hooks"""
        return self.hooksmap.get(cls, [])  # type: ignore

    def call_hooks(self, cls: Type, method: Callable, *args, **kwargs):
        for hook in self.hooks(cls):
            method(hook, *args, **kwargs)
