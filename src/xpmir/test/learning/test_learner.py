from pathlib import Path
from typing import Dict, Iterator
import torch
from experimaestro import DirectoryContext
from xpmir.learning import Random
from xpmir.learning.learner import Learner, Module
from xpmir.learning.optim import Adam, get_optimizers
from xpmir.learning.trainers import Trainer
from experimaestro.taskglobals import Env as TaskEnv


class TestModel(Module):
    def __post_init__(self):
        self.layer = torch.nn.Linear(3, 5)

    def initialize(self, random):
        pass


class TestTrainer(Trainer):
    def __post_init__(self):
        super().__post_init__()
        self.batches = []

    def iter_batches(self) -> Iterator:
        i = 0
        while True:
            yield [i, i + 1]
            i += 2

    def process_batch(self, batch):
        self.batches.extend(batch)

    def load_state_dict(self, state: Dict):
        raise NotImplementedError()

    def state_dict(self):
        raise NotImplementedError()


def test_learner(tmp_path: Path):
    model = TestModel()
    trainer = TestTrainer()

    learner = Learner(
        random=Random(),
        model=model,
        max_epochs=10,
        use_fp16=True,
        checkpoint_interval=1,
        trainer=trainer,
        optimizers=get_optimizers(Adam()),
        listeners=[],
        hooks=[],
    )

    TaskEnv.taskpath = tmp_path
    context = DirectoryContext(tmp_path)
    learner.instance(context=context).execute()