import logging
import json
from pathlib import Path
from typing import Dict, Iterator
from datamaestro_text.data.ir import Adhoc
from experimaestro import Param, pathgenerator, Annotated
import numpy as np
from xpmir.utils.utils import easylog
from xpmir.evaluation import evaluate
from xpmir.learning.context import (
    TrainState,
    TrainerContext,
)
from xpmir.rankers import (
    Retriever,
)
from xpmir.learning.learner import LearnerListener, Learner, LearnerListenerStatus
from xpmir.learning.optim import ModuleLoader

logger = easylog()


class ValidationListener(LearnerListener):
    """Learning validation early-stopping

    Computes a validation metric and stores the best result. If early_stop is
    set (> 0), then it signals to the learner that the learning process can
    stop.
    """

    metrics: Param[Dict[str, bool]] = {"map": True}
    """Dictionary whose keys are the metrics to record, and boolean
            values whether the best performance checkpoint should be kept for
            the associated metric ([parseable by ir-measures](https://ir-measur.es/))"""

    dataset: Param[Adhoc]
    """The dataset to use"""

    retriever: Param[Retriever]
    """The retriever for validation"""

    warmup: Param[int] = -1
    """How many epochs before actually computing the metric"""

    bestpath: Annotated[Path, pathgenerator("best")]
    """Path to the best checkpoints"""

    info: Annotated[Path, pathgenerator("info.json")]
    """Path to the JSON file that contains the metric values at each epoch"""

    validation_interval: Param[int] = 1
    """Epochs between each validation"""

    early_stop: Param[int] = 0
    """Number of epochs without improvement after which we stop learning.
    Should be a multiple of validation_interval or 0 (no early stopping)"""

    def __validate__(self):
        assert (
            self.early_stop % self.validation_interval == 0
        ), "Early stop should be a multiple of the validation interval"

    def initialize(self, learner: Learner, context: TrainerContext):
        super().initialize(learner, context)

        self.retriever.initialize()
        self.bestpath.mkdir(exist_ok=True, parents=True)

        # Checkpoint start
        try:
            with self.info.open("rt") as fp:
                self.top = json.load(fp)  # type: Dict[str, Dict[str, float]]
        except Exception:
            self.top = {}

    def update_metrics(self, metrics: Dict[str, float]):
        if self.top:
            # Just use another key
            for metric in self.metrics.keys():
                metrics[f"{self.id}/final/{metric}"] = self.top[metric]["value"]

    def monitored(self) -> Iterator[str]:
        return [key for key, monitored in self.metrics.items() if monitored]

    def task_outputs(self, learner: "Learner", dep):
        """Experimaestro outputs: returns the best checkpoints for each
        metric"""
        res = {
            key: ModuleLoader.construct(
                learner.model, self.bestpath / key / TrainState.MODEL_PATH, dep
            )
            for key, store in self.metrics.items()
            if store
        }

        return res

    def init_task(self, learner: "Learner", dep):
        return {
            key: dep(
                ModuleLoader(
                    value=learner.model,
                    path=self.bestpath / key / TrainState.MODEL_PATH,
                )
            )
            for key, store in self.metrics.items()
            if store
        }

    def should_stop(self, epoch=0):
        if self.early_stop > 0 and self.top:
            epochs_since_imp = (epoch or self.context.epoch) - max(
                info["epoch"] for key, info in self.top.items() if self.metrics[key]
            )
            if epochs_since_imp >= self.early_stop:
                return LearnerListenerStatus.STOP

        return LearnerListenerStatus.DONT_STOP

    def __call__(self, state: TrainState):
        # Check that we did not stop earlier (when loading from checkpoint / if other
        # listeners have not stopped yet)
        if self.should_stop(state.epoch - 1) == LearnerListenerStatus.STOP:
            return LearnerListenerStatus.STOP

        if state.epoch % self.validation_interval == 0:
            # Compute validation metrics
            means, details = evaluate(
                self.retriever, self.dataset, list(self.metrics.keys()), True
            )

            for metric, keep in self.metrics.items():
                value = means[metric]

                self.context.writer.add_scalar(
                    f"{self.id}/{metric}/mean", value, state.step
                )

                self.context.writer.add_histogram(
                    f"{self.id}/{metric}",
                    np.array(list(details[metric].values()), dtype=np.float32),
                    state.step,
                )

                # Update the top validation
                if state.epoch >= self.warmup:
                    topstate = self.top.get(metric, None)
                    if topstate is None or value > topstate["value"]:
                        # Save the new top JSON
                        self.top[metric] = {"value": value, "epoch": self.context.epoch}

                        # Copy in corresponding directory
                        if keep:
                            logging.info(
                                f"Saving the checkpoint {state.epoch}"
                                f" for metric {metric}"
                            )
                            self.context.copy(self.bestpath / metric)

            # Update information
            with self.info.open("wt") as fp:
                json.dump(self.top, fp)

        # Early stopping?
        return self.should_stop()
