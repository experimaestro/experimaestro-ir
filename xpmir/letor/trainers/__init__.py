import json
import os
from pathlib import Path
from shutil import rmtree
from typing import Dict, Iterator, Optional
from experimaestro import Option, Config, Param
from experimaestro import tqdm
from experimaestro.utils import cleanupdir
import torch
from torch.utils.tensorboard import SummaryWriter
from xpmir.letor.samplers import Sampler
from xpmir.letor.schedulers import Scheduler
from xpmir.letor.records import BaseRecords
from xpmir.rankers import LearnableScorer
from xpmir.utils import EasyLogger, easylog
from xpmir.letor.optim import Adam, Optimizer
from xpmir.letor import Device, DEFAULT_DEVICE
from xpmir.letor.batchers import Batcher
from xpmir.letor.metrics import MetricAccumulator

logger = easylog()


class TrainState:
    """Represents the state to be saved"""

    def __init__(self, state: "TrainState" = None):
        # Model and optimizer
        self.ranker = (
            state.ranker if state else None
        )  # type: Optional["LearnableScorer"]

        self.optimizer = state.optimizer if state else None
        self.scheduler = state.scheduler if state else None

        # The epoch
        self.epoch = state.epoch if state else 0

        # Was it loaded from disk?
        self.cached = False

        # Was it saved?
        self.path = None

    def __getstate__(self):
        return {"epoch": self.epoch}

    def save(self, path):
        """Save the state"""
        cleanupdir(path)

        with (path / "info.json").open("wt") as fp:
            json.dump(self.__getstate__(), fp)

        torch.save(
            {
                "model_state_dict": self.ranker.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path / "checkpoint.pth",
        )

        # FIXME: should try to serialize the sampler

        self.path = path

    def load(self, path, onlyinfo=False):
        if not onlyinfo:
            checkpoint = torch.load(path / "checkpoint.pth")
            assert self.ranker is not None
            self.ranker.load_state_dict(checkpoint["model_state_dict"])

            if self.optimizer is not None:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        with (path / "info.json").open("rt") as fp:
            self.__dict__.update(json.load(fp))

        # FIXME: should try to unserialize the sampler

        self.path = path
        self.cached = True


class TrainContext(EasyLogger):
    """Contains all the information about the training context"""

    PREFIX = "epoch-"
    STATETYPE = TrainState

    def __init__(self, logpath: Path, path: Path):
        self.path = path
        self.state = self.newstate()
        self.logpath = logpath
        self._writer = None

    @property
    def writer(self):
        """Returns a tensorboard writer

        by default, purges the entries beside the current epoch
        """
        if self._writer is None:
            self._writer = SummaryWriter(self.logpath, purge_step=self.state.epoch)
        return self._writer

    @classmethod
    def newstate(cls, state=None):
        return cls.STATETYPE(state)

    @property
    def epoch(self):
        return self.state.epoch

    def nextepoch(self):
        self.oldstate = self.state
        self.state = self.newstate(self.oldstate)
        self.state.epoch += 1

    def load_bestcheckpoint(self, target):
        # Find all the potential epochs
        epochs = []
        for f in self.path.glob(f"{TrainContext.PREFIX}*"):
            epoch = int(f.name[len(TrainContext.PREFIX) :])
            if epoch <= target:
                epochs.append(epoch)
        epochs.sort(reverse=True)

        # Try to load the first one
        for epoch in epochs:
            self.logger.info("Loading from checkpoint at epoch %d", epoch)
            path = self.path / f"{TrainContext.PREFIX}{epoch:08d}"

            try:
                self.state = self.newstate()
                self.state.load(path)
                return True

            except Exception:
                rmtree(path)
                self.logger.exception(f"Cannot load from epoch %d", epoch)

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


class Trainer(Config, EasyLogger):
    """Generic trainer

    Attributes:

        sampler: The data sampler
        batcher: how to batch examples
    """

    sampler: Param[Sampler]
    optimizer: Param[Optimizer] = Adam()
    scheduler: Param[Optional[Scheduler]] = None
    device: Option[Device] = DEFAULT_DEVICE
    batch_size: Param[int] = 16
    batches_per_epoch: Param[int] = 128

    batcher: Param[Batcher] = Batcher()

    train_iter: Iterator[BaseRecords]

    def __validate__(self):
        return super().__validate__()

    def initialize(self, random, ranker: LearnableScorer, context: TrainContext):
        self.random = random
        self.ranker = ranker
        self.context = context
        self.writer = None
        self.sampler.initialize(random)

        self.logger.info(
            "Trainer: %d batches of size %d/epoch",
            self.batches_per_epoch,
            self.batch_size,
        )
        self.device = self.device(self.logger)

    def to(self, device):
        """Change the computing device (if this is needed)"""
        pass

    def iter_train(self, loadepoch: int) -> Iterator[TrainState]:
        context = self.context

        if self.context.load_bestcheckpoint(loadepoch):
            if self.scheduler:
                context.state.scheduler = self.scheduler(
                    context.state.optimizer,
                    last_epoch=loadepoch * self.batches_per_epoch,
                )
            yield context.state
        else:
            context.state.optimizer = self.optimizer(self.ranker.parameters())
            context.state.ranker = self.ranker
            if self.scheduler:
                context.state.scheduler = self.scheduler(context.state.optimizer)

        self.logger.info("Transfering model to device %s", self.device)
        context.state.ranker.to(self.device)
        self.to(self.device)
        b_count = self.batches_per_epoch * self.batch_size

        batcher = self.batcher.initialize(self.batch_size, self.do_train)

        while True:
            # Step to the next epoch
            context.nextepoch()

            # Put the model into training mode (just in case)
            context.state.ranker.train()

            # Train for an epoch
            with tqdm(
                leave=False, total=b_count, ncols=100, desc=f"train {context.epoch}"
            ) as pbar:

                # Epoch: loop over batches
                metrics = MetricAccumulator()
                for b in range(self.batches_per_epoch):
                    batch = next(self.train_iter)
                    metrics.merge(batcher(batch))
                    pbar.update(self.batch_size)

                    # Optimizer step and scheduler step
                    context.state.optimizer.step()
                    if context.state.scheduler:
                        context.state.scheduler.step()

            # Report metrics over the epoch
            metrics.report(self.context.epoch, self.context.writer, "train")

            if context.state.scheduler:
                for i, value in enumerate(context.state.scheduler.get_last_lr()):
                    self.context.writer.add_scalar(
                        f"train/scheduler_lr/{i}", value, self.context.epoch
                    )

            # Yields the current state (after one epoch)
            yield context.state

    def do_train(self, microbatches: Iterator[BaseRecords]):
        """Train on a series of microbatches"""
        metrics = MetricAccumulator()
        self.context.state.optimizer.zero_grad()
        for microbatch in microbatches:
            self.train_batch_backward(microbatch, metrics)
        return metrics

    def train_batch_backward(self, records: BaseRecords, metrics: MetricAccumulator):
        """Combines a batch train and backard"""
        loss = self.train_batch(records, metrics)
        loss.backward()
        return loss, metrics

    def train_batch(
        self, records: BaseRecords, metrics: MetricAccumulator
    ) -> torch.Tensor:
        raise NotImplementedError()
