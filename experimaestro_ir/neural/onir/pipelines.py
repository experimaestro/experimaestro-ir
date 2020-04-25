from experimaestro import task, argument

import numpy as np
from onir.log import Logger

from trainers import Trainer
from rankers import Ranker
from vocab import Vocab

from . import openNIR


@argument("max_epoch", type=int, default=1000)
@argument("early_stop", type=int, default=20)
@argument("warmup", type=int, default=-1)
@argument("val_metric", type=str, default="map")
@argument("purge_weights", type=bool, default=True)
@argument("test", type=bool, default=False)
@argument("initial_eval", type=bool, default=False)
@argument("skip_ds_init", type=bool, default=False)
@argument("only_cached", type=bool, default=False)
@argument("trainer", type=Trainer)
@argument("ranker", type=Ranker)
@argument("vocab", type=Vocab)
@task(openNIR.pipelines.learn)
class Learn:
    def execute(self):
        random = np.random.RandomState()
        logger = Logger("openNIR")

        train_ds = None  # TODO

        ranker = self.ranker.create(vocab, logger, random)
        trainer = self.trainer.create(ranker, vocab, train_ds, logger, random)

        for train_ctxt in trainer.iter_train(only_cached=self.only_cached):
            pass
