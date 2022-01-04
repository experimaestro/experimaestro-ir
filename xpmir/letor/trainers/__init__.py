from typing import Iterator, List, Optional
from experimaestro import Option, Config, Param, deprecate
from experimaestro import tqdm
import torch
import torch.nn as nn
import numpy as np
from xpmir.letor import schedulers
from xpmir.letor.samplers import Sampler
from xpmir.letor.schedulers import Scheduler
from xpmir.letor.records import BaseRecords
from xpmir.rankers import LearnableScorer
from xpmir.utils import EasyLogger, easylog
from xpmir.letor.optim import Adam, Module, Optimizer, ParameterOptimizer
from xpmir.letor import Device, DEFAULT_DEVICE
from xpmir.letor.batchers import Batcher
from xpmir.letor.context import (
    ScalarMetric,
    TrainingHook,
    TrainContext,
    TrainState,
    Metrics,
)
from xpmir.utils import foreach

logger = easylog()


class ParameterFilter:
    """Filters out already optimized parameters"""

    def __init__(self):
        self.parameters = set()

    def __call__(self, name, params):
        if params in self.parameters:
            return False
        self.parameters.add(params)
        return True


class ScheduledOptimizer:
    def __init__(
        self,
        param_optimizers: List[ParameterOptimizer],
        num_training_steps: int,
        module: Module,
        use_scaler: bool,
    ):
        self.schedulers = []
        self.scheduler_factories = []
        self.optimizers = []
        self.scheduler_steps = 0  # Number of scheduler steps\
        self.num_training_steps = num_training_steps

        filter = ParameterFilter()
        for param_optimizer in param_optimizers:
            optimizer = param_optimizer.create_optimizer(module, filter)
            self.optimizers.append(optimizer)
            self.scheduler_factories.append(param_optimizer.scheduler)
            if param_optimizer.scheduler is not None:
                self.schedulers.append(
                    param_optimizer.scheduler(
                        optimizer, num_training_steps, last_epoch=-1
                    )
                )
            else:
                self.schedulers.append(None)

        assert len(self.schedulers) == len(self.optimizers)

        if use_scaler:
            logger.info("Using GradScaler when optimizing")
        self.scaler = torch.cuda.amp.GradScaler() if use_scaler else None

    def load_state_dict(self, state):
        for optimizer, optimizer_state in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(optimizer_state)

        if self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])

        # Re-create schedulers
        self.steps = state["scheduler_steps"]
        for ix, scheduler_factory in enumerate(self.scheduler_factories):
            if scheduler_factory is not None:
                self.schedulers[ix] = scheduler_factory(
                    self.optimizers[ix],
                    self.num_training_steps,
                    last_epoch=self.scheduler_steps,
                )

    def state_dict(self):
        return {
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "scheduler_steps": self.scheduler_steps,
        }

    def scale(self, loss: torch.Tensor):
        if self.scaler is None:
            return loss
        return self.scaler.scale(loss)

    def zero_grad(self):
        """Zero-grad for all optimizers"""
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def optimizer_step(self, context: TrainContext):
        """Performs an optimizer step (using the scaler if defined)"""
        if self.scaler is None:
            for optimizer in self.optimizers:
                optimizer.step()

        else:
            for optimizer in self.optimizers:
                self.scaler.step(optimizer)
            context.metrics.add(
                ScalarMetric("train/gradient/scaler", self.scaler.get_scale(), 1)
            )
            self.scaler.update()

    def scheduler_step(self, context: TrainContext):
        for ix, scheduler in enumerate(self.schedulers):
            if scheduler is not None:
                for p_ix, lr in enumerate(scheduler.get_last_lr()):
                    context.metrics.add(
                        ScalarMetric(f"train/gradient/scheduler/{ix+1}/{p_ix+1}", lr, 1)
                    )
                scheduler.step()
        self.scheduler_steps += 1


class Trainer(Config, EasyLogger):
    """Generic trainer"""

    sampler: Param[Sampler]
    """The sampler to use"""

    optimizers: Param[List[ParameterOptimizer]]
    """The list of parameter optimizers"""

    device: Option[Device] = DEFAULT_DEVICE
    """The device(s) to be used for the model"""

    batch_size: Param[int] = 16
    """Number of samples per batch (the notion of sample depends on the sampler)"""

    batches_per_epoch: Param[int] = 128
    """Number of batches for one epoch (after each epoch results are reported)"""

    hooks: Param[List[TrainingHook]] = []
    """Hooks for this trainer: this includes the losses, but can be adapted for other uses
        The specific list of hooks depends on the specific trainer"""

    batcher: Param[Batcher] = Batcher()
    """How to batch samples together"""

    use_fp16: Param[bool] = False
    """Use mixed precision when training"""

    train_iter: Iterator[BaseRecords]

    def __validate__(self):
        assert self.optimizers, "At least one optimizer should be defined"
        return super().__validate__()

    def initialize(
        self,
        random: np.random.RandomState,
        ranker: LearnableScorer,
        context: TrainContext,
    ):
        self.random = random
        self.ranker = ranker
        self.context = context
        self.writer = None

        self.logger.info(
            "Trainer: %d batches of size %d/epoch",
            self.batches_per_epoch,
            self.batch_size,
        )
        self.device = self.device(self.logger)
        foreach(self.hooks, self.context.add_hook)

        self.sampler.initialize(random)

    def to(self, device):
        """Change the computing device (if this is needed)"""
        foreach(self.context.hooks(nn.Module), lambda hook: hook.to(device))

    def iter_train(self, max_epoch: int) -> Iterator[TrainState]:
        context = self.context

        self.logger.info("Transfering model to device %s", self.device)
        self.ranker.to(self.device)
        num_training_steps = context.state.epoch * self.batches_per_epoch
        optimizer = ScheduledOptimizer(
            self.optimizers, num_training_steps, self.ranker, self.use_fp16
        )

        if self.context.load_bestcheckpoint(
            max_epoch, self.ranker, optimizer, self.sampler
        ):
            yield context.state
        else:
            context.state.sampler = self.sampler
            context.state.optimizer = optimizer
            context.state.ranker = self.ranker

        self.to(self.device)
        b_count = self.batches_per_epoch * self.batch_size

        batcher = self.batcher.initialize(self.batch_size)

        assert context.state.optimizer is not None
        while True:
            # Step to the next epoch
            context.nextepoch()

            # Train for an epoch
            with tqdm(
                leave=False, total=b_count, ncols=100, desc=f"train {context.epoch}"
            ) as pbar:
                # Put the model into training mode (just in case)
                context.state.ranker.train()

                # Epoch: loop over batches
                metrics = Metrics()
                for b in range(self.batches_per_epoch):
                    batch = next(self.train_iter)
                    context.state.samplecount_add(len(batch))
                    self.context.nextbatch()
                    batcher.process_withreplay(batch, self.do_train)
                    pbar.update(self.batch_size)

                    # Optimizer step and scheduler step
                    context.state.optimizer.optimizer_step(context)
                    context.state.optimizer.scheduler_step(context)

                    # Update metrics
                    metrics.merge(context.metrics)

            # Report metrics over the epoch
            metrics.report(self.context.state.step, self.context.writer, "train")

            # Yields the current state (after one epoch)
            yield context.state

    def do_train(self, microbatches: Iterator[BaseRecords], length: int):
        """Train on a series of microbatches"""
        self.context.reset(True)
        self.context.state.optimizer.zero_grad()
        for ix, microbatch in enumerate(microbatches):
            self.train_batch_backward(microbatch)

    def add_losses(self, nrecords: int):
        """Add all losses"""
        total_loss = 0.0
        names = []

        for loss in self.context.losses:
            total_loss += loss.weight * loss.value
            names.append(loss.name)
            self.context.metrics.add(
                ScalarMetric(f"{loss.name}", float(loss.value.item()), nrecords)
            )

        # Reports the main metric
        names.sort()
        self.context.metrics.add(
            ScalarMetric("+".join(names), float(total_loss.item()), nrecords)
        )

        return total_loss

    def train_batch_backward(self, records: BaseRecords):
        """Combines a batch train and backard

        This method can be implemented by specific trainers that use the gradient.
        In that case the regularizer losses should be taken into account with
        `self.add_losses`.
        """
        self.context.reset(False)

        with torch.autocast(self.device.type, enabled=self.use_fp16):
            self.train_batch(records)
            loss = self.add_losses(len(records))
            self.context.state.optimizer.scale(loss).backward()
        return loss

    def train_batch(self, records: BaseRecords) -> torch.Tensor:
        raise NotImplementedError()
