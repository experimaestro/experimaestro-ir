from functools import partial, lru_cache
import logging

from xpmir.distributed import DistributedHook
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.schedulers import LinearWithWarmup
import xpmir.letor.trainers.pairwise as pairwise
from xpmir.neural.cross import CrossScorer
from experimaestro import experiment, setmeta
from experimaestro.launcherfinder import find_launcher
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.letor.optim import (
    AdamW,
    ParameterOptimizer,
    RegexParameterFilter,
    get_optimizers,
)
from xpmir.papers.cli import paper_command
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import DualTransformerEncoder
from xpmir.papers.results import PaperResults
from xpmir.papers.pipelines.msmarco import RerankerMSMarcoV1Experiment
from .configuration import Monobert, Learner as LearnerConfig

logging.basicConfig(level=logging.INFO)


class MonoBERTExperiment(RerankerMSMarcoV1Experiment):
    """MonoBERT experiment

    This class can be used for experiments that depend on the training of a
    monobert model
    """

    cfg: Monobert

    basemodel = BM25().tag("model", "bm25")

    def __init__(self, xp: experiment, cfg: Monobert):
        super().__init__(xp, cfg)
        self.launcher_learner = find_launcher(cfg.monobert.requirements)
        self.launcher_evaluate = find_launcher(cfg.retrieval.requirements)

    @lru_cache
    def get_optimizers(self, cfg: LearnerConfig):
        scheduler = (
            LinearWithWarmup(
                num_warmup_steps=cfg.num_warmup_steps,
                min_factor=cfg.warmup_min_factor,
            )
            if cfg.scheduler
            else None
        )

        return get_optimizers(
            [
                ParameterOptimizer(
                    scheduler=scheduler,
                    optimizer=AdamW(lr=cfg.lr, weight_decay=0, eps=1e-6),
                    filter=RegexParameterFilter(
                        includes=[r"\.bias$", r"\.LayerNorm\."]
                    ),
                ),
                ParameterOptimizer(
                    scheduler=scheduler,
                    optimizer=AdamW(lr=cfg.lr, weight_decay=1e-2, eps=1e-6),
                ),
            ]
        )

    def run(self) -> PaperResults:
        """monoBERT model"""

        cfg = self.cfg

        # Define the different launchers
        val_retrievers = partial(self.retrievers, k=cfg.monobert.validation_top_k)

        # define the trainer for monobert
        monobert_trainer = pairwise.PairwiseTrainer(
            lossfn=pairwise.PointwiseCrossEntropyLoss(),
            sampler=self.train_sampler,
            batcher=PowerAdaptativeBatcher(),
            batch_size=cfg.monobert.batch_size,
        )

        monobert_scorer: CrossScorer = CrossScorer(
            encoder=DualTransformerEncoder(
                model_id="bert-base-uncased", trainable=True, maxlen=512, dropout=0.1
            )
        ).tag("reranker", "monobert")

        # The validation listener evaluates the full retriever
        # (retriever + reranker) and keep the best performing model
        # on the validation set
        validation = ValidationListener(
            dataset=self.ds_val,
            retriever=self.model_based_retrievers(
                self.documents,
                retrievers=val_retrievers,
                scorer=monobert_scorer,
                device=self.device,
            ),
            validation_interval=cfg.monobert.validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG": False},
        )

        # The learner trains the model
        learner = Learner(
            # Misc settings
            device=self.device,
            random=self.random,
            # How to train the model
            trainer=monobert_trainer,
            # The model to train
            scorer=monobert_scorer,
            # Optimization settings
            steps_per_epoch=cfg.monobert.steps_per_epoch,
            optimizers=self.get_optimizers(cfg.monobert),
            max_epochs=cfg.monobert.max_epochs,
            # The listeners (here, for validation)
            listeners={"bestval": validation},
            # The hook used for evaluation
            hooks=[setmeta(DistributedHook(models=[monobert_scorer]), True)],
        )

        # Submit job and link
        outputs = learner.submit(launcher=self.launcher_learner)
        self.tb.add(learner, learner.logpath)

        # Evaluate the neural model on test collections
        for metric_name in validation.monitored():
            model = outputs.listeners["bestval"][metric_name]  # type: CrossScorer
            self.tests.evaluate_retriever(
                partial(
                    self.model_based_retrievers,
                    scorer=model,
                    retrievers=self.test_retrievers,
                    device=self.device,
                ),
                self.launcher_evaluate,
                model_id=f"monobert-{metric_name}",
            )

        return PaperResults(
            models={"monobert-RR@10": outputs.listeners["bestval"]["RR@10"]},
            evaluations=self.tests,
            tb_logs={"monobert-RR@10": learner.logpath},
        )


@paper_command(schema=Monobert, package=__package__)
def cli(xp: experiment, cfg: Monobert):
    return MonoBERTExperiment(xp, cfg).run()
