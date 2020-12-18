import json
import os
from datamaestro_text.data.ir import Adhoc
from experimaestro import task, param, progress, pathoption
from xpmir.letor import Random
from xpmir.letor.samplers import Sampler
from xpmir.letor.trainers import Trainer
from xpmir.rankers import Retriever
from xpmir.utils import logger


@param("max_epoch", default=1000, help="Maximum training epoch")
@param(
    "early_stop", default=20, help="Maximum number of epochs without improvement (val)"
)
@param("warmup", default=-1, help="Number of warmup epochs")
@param("purge_weights", default=True)
@param("initial_eval", default=False)
@param("only_cached", default=False)
@param("val_metric", default="map")
@param("val_dataset", type=Adhoc)
@param("ranker", type=Retriever)
@param("sampler", type=Sampler, help="Training data sampler")
@param("ranker", type=Retriever)
@param("trainer", type=Trainer)
@param("random", type=Random)
@pathoption("predictor_path", "predictor")
@pathoption("valtest_path", "val_test.jsonl")
@task()
class Learner:
    """Learns a ranker"""

    def execute(self):
        self.logger = logger()
        self.ranker.initialize(self.random.state)
        self.trainer.initialize(self.random.state, self.ranker, self.sampler)
        self.valid_pred.initialize(
            self.predictor_path,
            [self.val_metric],
            self.random.state,
            self.ranker,
            self.val_dataset,
        )
        self.sampler.initialize(self.ranker.vocab)
        self.val_dataset.initialize(self.ranker.vocab)

        validator = self.valid_pred.pred_ctxt()

        top_epoch, top_value, top_train_ctxt, top_valid_ctxt = None, None, None, None
        prev_train_ctxt = None

        file_output = {"validation_metric": self.val_metric}

        for train_ctxt in self.trainer.iter_train(only_cached=self.only_cached):
            # Report progress
            progress(train_ctxt["epoch"] / self.max_epoch)

            if (
                prev_train_ctxt is not None
                and top_epoch is not None
                and prev_train_ctxt is not top_train_ctxt
            ):
                self._purge_weights(prev_train_ctxt)

            if train_ctxt["epoch"] >= 0 and not self.only_cached:
                message = self._build_train_msg(train_ctxt)

                if train_ctxt["cached"]:
                    self.logger.debug(f"[train] [cached] {message}")
                else:
                    self.logger.debug(f"[train] {message}")

            if train_ctxt["epoch"] == -1 and not self.initial_eval:
                continue

            # Compute validation metrics
            valid_ctxt = dict(validator(train_ctxt))

            message = self._build_valid_msg(valid_ctxt)

            if valid_ctxt["epoch"] >= self.warmup:
                if self.val_metric == "":
                    top_epoch = valid_ctxt["epoch"]
                    top_train_ctxt = train_ctxt
                    top_valid_ctxt = valid_ctxt
                elif (
                    top_value is None
                    or valid_ctxt["metrics"][self.val_metric] > top_value
                ):
                    message += " <---"
                    top_epoch = valid_ctxt["epoch"]
                    top_value = valid_ctxt["metrics"][self.val_metric]
                    if top_train_ctxt is not None:
                        self._purge_weights(top_train_ctxt)
                    top_train_ctxt = train_ctxt
                    top_valid_ctxt = valid_ctxt
            else:
                if prev_train_ctxt is not None:
                    self._purge_weights(prev_train_ctxt)

            if not self.only_cached:
                if valid_ctxt["cached"]:
                    self.logger.debug(f"[valid] [cached] {message}")
                else:
                    self.logger.info(f"[valid] {message}")

            if top_epoch is not None:
                epochs_since_imp = valid_ctxt["epoch"] - top_epoch
                if self.early_stop > 0 and epochs_since_imp >= self.early_stop:
                    self.logger.warn(
                        "stopping after epoch {epoch} ({early_stop} epochs with no "
                        "improvement to {val_metric})".format(
                            **valid_ctxt, **self.__dict__
                        )
                    )
                    break

            if train_ctxt["epoch"] >= self.max_epoch:
                self.logger.warn(
                    "stopping after epoch {max_epoch} (max_epoch)".format(
                        **self.__dict__
                    )
                )
                break

            prev_train_ctxt = train_ctxt

        self.logger.info(
            "top validation epoch={} {}={}".format(
                top_epoch, self.val_metric, top_value
            )
        )

        file_output.update(
            {
                "valid_epoch": top_epoch,
                "valid_run": top_valid_ctxt["run_path"],
                "valid_path": top_train_ctxt["ranker_path"],
                "valid_metrics": top_valid_ctxt["metrics"],
            }
        )

        with open(self.valtest_path, "wt") as f:
            json.dump(file_output, f)
            f.write("\n")

        self.logger.info("valid run at {}".format(valid_ctxt["run_path"]))
        self.logger.info("valid " + self._build_valid_msg(top_valid_ctxt))

    def _build_train_msg(self, ctxt):
        delta_acc = ctxt["acc"] - ctxt["unsup_acc"]
        msg_pt1 = "epoch={epoch} loss={loss:.4f}".format(**ctxt)
        msg_pt2 = (
            "acc={acc:.4f} unsup_acc={unsup_acc:.4f} "
            "delta_acc={delta_acc:.4f}".format(**ctxt, delta_acc=delta_acc)
        )
        losses = ""
        if ctxt["losses"] and (
            {"data"} != ctxt["losses"].keys() or ctxt["losses"]["data"] != ctxt["loss"]
        ):
            losses = []
            for lname, lvalue in ctxt["losses"].items():
                losses.append(f"{lname}={lvalue:.4f}")
            losses = " ".join(losses)
            losses = f" ({losses})"
        return f"{msg_pt1}{losses} {msg_pt2}"

    def _build_valid_msg(self, ctxt):
        message = ["epoch=" + str(ctxt["epoch"])]
        for metric, value in sorted(ctxt["metrics"].items()):
            message.append("{}={:.4f}".format(metric, value))
            if metric == self.val_metric:
                message[-1] = "[" + message[-1] + "]"
        return " ".join(message)

    def _purge_weights(self, ctxt):
        if self.purge_weights:
            if os.path.exists(ctxt["ranker_path"]):
                os.remove(ctxt["ranker_path"])
            if os.path.exists(ctxt["optimizer_path"]):
                os.remove(ctxt["optimizer_path"])
