from pathlib import Path
import json
import torch

from experimaestro import Param, Annotated, pathgenerator, Meta, tqdm

from xpmir.neural.generative import ConditionalGenerator
from xpmir.learning.batchers import Batcher
from xpmir.learning.optim import ModuleInitMode, ModuleLoader
from xpmir.neural.generative.referential.samplers import (
    ReferentialSampler,
    ReferentialDocumentIDRecords,
)
from xpmir.learning.context import TrainState, TrainerContext
from xpmir.learning.learner import LearnerListener, Learner, LearnerListenerStatus
from xpmir.utils.utils import easylog, batchiter

logger = easylog()


class ReferentialCrossEntropyLossListener(LearnerListener):
    """A listener based on the result of the cross entropy loss"""

    max_depth: Param[int] = 5
    """The max depth of the model"""

    batch_size: Meta[int] = 128

    batcher: Meta[Batcher] = Batcher()
    """The way to prepare batches of documents"""

    sampler: Param[ReferentialSampler]
    """A sampler which provide a iterator over the validation dataset"""

    model: Param[ConditionalGenerator]
    """The conditional generator for generating the proba"""

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

    validation_size: Param[int] = 1000
    """Just for logging use"""

    def __validate__(self):
        assert (
            self.early_stop % self.validation_interval == 0
        ), "Early stop should be a multiple of the validation interval"

    def initialize(self, learner: Learner, context: TrainerContext):
        super().initialize(learner, context)
        self.model.initialize(ModuleInitMode.DEFAULT.to_options(learner.random.state))
        self.sampler.initialize(learner.random.state)
        self.bestpath.mkdir(exist_ok=True, parents=True)
        # Checkpoint start
        try:
            with self.info.open("rt") as fp:
                self.top = json.load(fp)  # type: Dict[str, Dict[str, float]]
        except Exception:
            self.top = {}

    def task_outputs(self, learner: "Learner", dep):
        """Experimaestro outputs: returns the best checkpoints for each
        metric"""
        res = {
            "CE": ModuleLoader.construct(
                learner.model, self.bestpath / "CE" / TrainState.MODEL_PATH, dep
            )
        }

        return res

    def init_task(self, learner: "Learner", dep):
        return {
            "CE": dep(
                ModuleLoader(
                    value=learner.model,
                    path=self.bestpath / "CE" / TrainState.MODEL_PATH,
                )
            )
        }

    def should_update_validation(self, state: TrainState) -> bool:
        return state.epoch >= self.warmup

    def compute(self, records, loss):
        sequence_generator = self.model.sequence_generator()
        eos_token_id = self.model.eos_token_id
        posdocs_text = [doc.document.get_text() for doc in records.documents]

        # prepare the label, add and eos and then padding it with -100.
        max_length = max(len(target) for target in records.targets)
        if max_length < self.max_depth:
            target_matrix_width = max_length + 1
        else:
            target_matrix_width = self.max_depth

        padded_labels = []
        for label in records.targets:
            tmp_count = len(label)
            if tmp_count < target_matrix_width:
                # -100 is the sign for not treating this lable for CE loss
                label = (
                    label
                    + [eos_token_id]
                    + [-100] * (target_matrix_width - tmp_count - 1)
                )
            padded_labels.append(label)
        padded_labels = torch.tensor(padded_labels).to(self.model.device)
        sequence_generator.init(posdocs_text)
        loss.append(sequence_generator.decode(labels=padded_labels).loss)

    def __call__(self, state: TrainState) -> LearnerListenerStatus:
        # the early stop is not implemented yet
        if state.epoch % self.validation_interval == 0:
            self.model.eval()
            loss = []  # a list to contain the sum of loss for each batch
            batcher = self.batcher.initialize(self.batch_size)
            doc_iter = tqdm(
                self.sampler.referential_iter(),
                total=self.validation_size,
                desc="Validation",
            )
            for batch in batchiter(self.batch_size, doc_iter):
                records = ReferentialDocumentIDRecords()
                for record in batch:
                    records.add(record)
                with torch.no_grad():
                    batcher.process(records, self.compute, loss)
            value = float(torch.sum(torch.stack(loss, dim=0), dim=0))

            self.context.writer.add_scalar(f"{self.id}/CE/mean", value, state.step)

            if self.should_update_validation(state):
                topstate = self.top.get("CE", None)
                # for ce loss, the smaller, the better
                if topstate is None or value < topstate["value"]:
                    # Save the new top JSON
                    self.top["CE"] = {"value": value, "epoch": self.context.epoch}

                    # Copy in corresponding directory
                    logger.info(f"Saving the checkpoint {state.epoch} for metric CE")
                    self.context.copy(self.bestpath / "CE")

            # Update information
            with self.info.open("wt") as fp:
                json.dump(self.top, fp)

        return LearnerListenerStatus.DONT_STOP
