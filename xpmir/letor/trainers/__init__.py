from pathlib import Path
from experimaestro import option, param, pathoption, config
from experimaestro.tqdm import tqdm
import torch
from xpmir.letor.samplers import Sampler
from xpmir.utils import easylog
from xpmir.letor.optim import Adam, Optimizer
from xpmir.letor import Device, DEFAULT_DEVICE


class TrainContext:
    def __init__(self, path: Path):
        self.path = path
        self.epoch = -1
        self.cached = False

    def load_checkpoint(self):
        pass

    def save_checkpoint(self):
        pass


@param("sampler", type=Sampler, help="Training data sampler")
@param("batch_size", default=16)
@param("batches_per_epoch", default=32)
@option(
    "grad_acc_batch",
    default=0,
    help="With memory issues, this parameter can be used to give the size of a micro-batch",
)
@param("optimizer", type=Optimizer, default=Adam())
@option("device", type=Device, default=DEFAULT_DEVICE)
@pathoption("checkpointspath", "checkpoints")
@config()
class Trainer:
    def initialize(self, random, ranker):
        self.random = random
        self.ranker = ranker

        self.logger = easylog()
        if self.grad_acc_batch > 0:
            assert (
                self.batch_size % self.grad_acc_batch == 0
            ), "batch_size must be a multiple of grad_acc_batch"
            self.num_microbatches = self.batch_size // self.grad_acc_batch
            self.batch_size = self.grad_acc_batch
        else:
            self.num_microbatches = 1
        self.device = self.device(self.logger)

    def iter_train(self, only_cached=False):
        context = TrainContext(self.checkpointspath)

        b_count = self.batches_per_epoch * self.num_microbatches * self.batch_size

        ranker = self.ranker.to(self.device)
        optimizer = self.optimizer(self.ranker.parameters())

        yield context  # before training

        while True:
            context.epoch += 1
            if context.load_checkpoint():
                continue

            # forward to previous versions (if needed)
            ranker.train()

            with tqdm(
                leave=False, total=b_count, ncols=100, desc=f"train {context.epoch}"
            ) as pbar:
                for b in range(self.batches_per_epoch):
                    for _ in range(self.num_microbatches):
                        loss = self.train_batch()
                        loss.backward()
                        pbar.update(self.batch_size)

                    optimizer.step()
                    optimizer.zero_grad()

            yield context

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
