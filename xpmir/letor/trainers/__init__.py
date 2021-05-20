import json
import os
from pathlib import Path
from shutil import rmtree
from typing import Dict, Iterator
from experimaestro import Option, Config, help, Param
from experimaestro import tqdm
from experimaestro.utils import cleanupdir
import torch
from torch.utils.tensorboard import SummaryWriter
from typing_extensions import Annotated
from xpmir.letor.samplers import Sampler
from xpmir.utils import EasyLogger, easylog
from xpmir.letor.optim import Adam, Optimizer
from xpmir.letor import Device, DEFAULT_DEVICE


class TrainState:
    """Represents the state to be saved"""

    def __init__(self, state: "TrainState" = None):
        # Model and optimizer
        self.ranker = state.ranker if state else None
        self.optimizer = state.optimizer if state else None

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

        with (path / "checkpoint.pth").open("wb") as fp:
            torch.save((self.ranker, self.optimizer), fp)

        self.path = path

    def load(self, path, onlyinfo=False):
        if not onlyinfo:
            with (path / "checkpoint.pth").open("rb") as fp:
                self.ranker, self.optimizer = torch.load(fp)

        with (path / "info.json").open("rt") as fp:
            self.__dict__.update(json.load(fp))

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
    """

    sampler: Param[Sampler]
    optimizer: Param[Optimizer] = Adam()
    device: Option[Device] = DEFAULT_DEVICE
    batch_size: Param[int] = 16
    batches_per_epoch: Param[int] = 128
    # How to split batches if memory issues
    grad_acc_batch: Param[int] = 0

    def initialize(self, random, ranker, context: TrainContext):
        self.random = random
        self.ranker = ranker
        self.context = context
        self.writer = None
        self.sampler.initialize(random)

        if self.grad_acc_batch > 0:
            assert (
                self.batch_size % self.grad_acc_batch == 0
            ), "batch_size must be a multiple of grad_acc_batch"
            self.num_microbatches = self.batch_size // self.grad_acc_batch
            self.batch_size = self.grad_acc_batch
        else:
            self.num_microbatches = 1

        self.logger.info(
            "Trainer: batch %d x %d (micro-batches), %d batches/epoch",
            self.batch_size,
            self.num_microbatches,
            self.batches_per_epoch,
        )
        self.device = self.device(self.logger)

    def iter_train(self, loadepoch: int) -> Iterator[TrainState]:
        context = self.context

        if not self.context.load_bestcheckpoint(loadepoch):
            context.state.optimizer = self.optimizer(self.ranker.parameters())
            context.state.ranker = self.ranker
        else:
            yield context.state

        self.logger.info("Transfering model to device %s", self.device)
        context.state.ranker.to(self.device)
        b_count = self.batches_per_epoch * self.num_microbatches * self.batch_size

        while True:
            context.nextepoch()

            # forward to previous versions (if needed)
            context.state.ranker.train()

            with tqdm(
                leave=False, total=b_count, ncols=100, desc=f"train {context.epoch}"
            ) as pbar:
                total_metrics = {}
                for b in range(self.batches_per_epoch):
                    for _ in range(self.num_microbatches):
                        loss, metrics = self.train_batch()
                        loss.backward()
                        total_metrics = {
                            key: value + total_metrics.get(key, 0.0)
                            for key, value in metrics.items()
                        }
                        pbar.update(self.batch_size)

                    context.state.optimizer.step()
                    context.state.optimizer.zero_grad()

            for key, value in total_metrics.items():
                self.context.writer.add_scalar(
                    f"train/{key}",
                    value / (self.num_microbatches * self.batches_per_epoch),
                    self.context.epoch,
                )

            yield context.state

    def train_batch(self):
        raise NotImplementedError()

    def fast_forward(self, record_count):
        raise NotImplementedError()

    def _fast_forward(self, train_it, fields, record_count):
        # Since the train_it holds a refernece to fields, we can greatly speed up the "fast forward"
        # process by temporarily clearing the requested fields (meaning that the train iterator
        # should simply find the records/pairs to return, but yield an empty payload).
        orig_fields = set(fields)
        try:
            fields.clear()
            for _ in zip(range(record_count), train_it):
                pass
        finally:
            fields.update(orig_fields)
