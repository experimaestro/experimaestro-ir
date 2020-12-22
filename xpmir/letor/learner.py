import json
from pathlib import Path
from typing import Dict
from datamaestro_text.data.ir import Adhoc
from experimaestro import task, config, param, progress, pathoption
from xpmir.evaluation import evaluate
from xpmir.letor import Random
from xpmir.letor.samplers import Sampler
from xpmir.letor.trainers import TrainContext, Trainer
from xpmir.rankers import LearnableScorer, Retriever, Scorer
from xpmir.utils import easylog


class ValidationContext:
    # Train context
    train_ctxt: TrainContext

    # All metrics
    metrics: Dict[str, float]

    # Validation value
    value: float

    @property
    def epoch(self):
        return self.train_ctxt.epoch

    def __init__(self, train_ctxt, metrics, value):
        self.train_ctxt = train_ctxt
        self.metrics = metrics
        self.value = value


@param("metric", default="map")
@param("dataset", type=Adhoc)
@param("retriever", type=Retriever)
@config()
class Validation:
    def initialize(self, path: Path):
        self.path = path
        self.retriever.initialize()
        self.metrics = [self.metric]

    def __call__(self, train_ctxt: TrainContext) -> ValidationContext:
        # Evaluate
        mean, by_query = evaluate(
            self.path / "run.trec", self.retriever, self.dataset, self.metrics
        )
        return ValidationContext(train_ctxt, mean, mean[self.metric])


# Training
@param("max_epoch", default=1000, help="Maximum training epoch")
@param(
    "early_stop",
    default=20,
    help="Maximum number of epochs without improvement on validation",
)
@param("warmup", default=-1, help="Number of warmup epochs")
@param("purge_weights", default=True)
@param("random", type=Random, help="Random generator")
@param("initial_eval", default=False)
@param("trainer", type=Trainer, help="The trainer used to learn the parameters")
@param("scorer", type=LearnableScorer, help="The scorer to learn")

# Validation
@param("validation", type=Validation, help="How to compute the validation metric")
@pathoption("modelpath", "model")
@task()
class Learner(Scorer):
    trainer: Trainer

    """The learner task is generic, and takes two main arguments:
    (1) the scorer defines the model (e.g. DRMM), and
    (2) the trainer defines the loss (e.g. pointwise, pairwise, etc.)
    """

    def execute(self):
        self.logger = easylog()

        self.only_cached = False
        self.modelpath.mkdir(exist_ok=True, parents=True)

        # Initialize the scorer and trainer
        self.scorer.initialize(self.random.state)
        self.trainer.initialize(self.random.state, self.scorer)
        self.validation.initialize(self.modelpath)

        # Top validation context
        top = None

        # Previous train context
        prev_train_ctxt = None

        for train_ctxt in self.trainer.iter_train():
            # Report progress
            progress(train_ctxt.epoch / self.max_epoch)

            if train_ctxt.epoch >= 0 and not self.only_cached:
                message = f"epoch {train_ctxt.epoch}"
                if train_ctxt.cached:
                    self.logger.debug(f"[train] [cached] {message}")
                else:
                    self.logger.debug(f"[train] {message}")

            if train_ctxt.epoch == -1 and not self.initial_eval:
                continue

            # Compute validation metrics
            valid_ctxt = self.validation(train_ctxt)

            # Update the top validation
            if valid_ctxt.epoch >= self.warmup:
                if top is None or valid_ctxt.value > top.value:
                    top = valid_ctxt
                    valid_ctxt.save()

            # Early stopping
            if top is not None:
                epochs_since_imp = valid_ctxt.epoch - top.epoch
                if self.early_stop > 0 and epochs_since_imp >= self.early_stop:
                    self.logger.warn(
                        "stopping after epoch {epoch} ({early_stop} epochs with no "
                        "improvement to {val_metric})".format(
                            **valid_ctxt, **self.__dict__
                        )
                    )
                    break

            # Early stopping
            if train_ctxt.epoch >= self.max_epoch:
                self.logger.warn(
                    "stopping after epoch {max_epoch} (max_epoch)".format(
                        **self.__dict__
                    )
                )
                break

        self.logger.info("top validation epoch={} {}".format(top.epoch, top.value))
