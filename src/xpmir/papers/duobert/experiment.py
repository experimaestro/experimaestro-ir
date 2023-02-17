# Implementation of the experiments in the paper
# Multi-Stage Document Ranking with BERT (Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, Jimmy Lin). 2019.
# https://arxiv.org/abs/1910.14424

# An imitation of examples/msmarco-reranking.py

# flake8: noqa: T201

from functools import partial
import logging
from xpmir.distributed import DistributedHook
from xpmir.letor.learner import Learner, ValidationListener

import xpmir.letor.trainers.pairwise as pairwise
from xpmir.neural.cross import CrossScorer, DuoCrossScorer
from experimaestro import experiment, setmeta
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.letor.optim import (
    get_optimizers,
)
from xpmir.papers.cli import paper_command
from xpmir.text.huggingface import DualDuoBertTransformerEncoder
from xpmir.papers.monobert.experiment import MonoBERTExperiment
from xpmir.papers.results import PaperResults
from .configuration import DuoBERT

logging.basicConfig(level=logging.INFO)


class DuoBERTExperiment(MonoBERTExperiment):
    cfg: DuoBERT

    def run(self):
        """Runs an experiment"""
        # Run monobert
        cfg = self.cfg
        monobert_results = super().run()
        monobert_scorer = monobert_results.models["monobert-RR@10"]

        monobert_retrievers = partial(
            self.model_based_retrievers,
            scorer=monobert_scorer,
            retrievers=self.test_retrievers,
            device=self.device,
        )

        monobert_based_retrievers = partial(
            monobert_retrievers,
            batch_size=cfg.retrieval.batch_size,
            batcher=PowerAdaptativeBatcher(),
            device=self.device,
        )

        monobert_val_retrievers = partial(
            monobert_based_retrievers, k=cfg.monobert.validation_top_k
        )
        val_retrievers = partial(
            monobert_val_retrievers, k=cfg.duobert.validation_top_k
        )

        monobert_test_retrievers = partial(
            monobert_retrievers, k=cfg.monobert.test_top_k
        )
        test_retrievers = partial(monobert_test_retrievers, k=cfg.duobert.test_top_k)

        # ------Start the code for the duobert

        # Define the trainer for the duobert
        duobert_trainer = pairwise.DuoPairwiseTrainer(
            lossfn=pairwise.DuoLogProbaLoss().tag("loss", "duo_logp"),
            sampler=self.train_sampler,
            batcher=PowerAdaptativeBatcher(),
            batch_size=cfg.duobert_learner.batch_size,
        )

        # The scorer(model) for the duobert
        duobert_scorer: DuoCrossScorer = DuoCrossScorer(
            encoder=DualDuoBertTransformerEncoder(trainable=True, dropout=0.1)
        ).tag("duo-reranker", "duobert")

        duobert_validation = ValidationListener(
            dataset=self.ds_val,
            retriever=duobert_scorer.getRetriever(
                val_retrievers(self.documents),
                cfg.duobert.batch_size,
                PowerAdaptativeBatcher(),
                device=self.device,
            ),
            validation_interval=cfg.duobert.validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG": False},
        )

        # The validation listener evaluates the full retriever
        # (retriever + reranker) and keep the best performing model
        # on the validation set
        validation = ValidationListener(
            dataset=self.ds_val,
            retriever=self.model_based_retrievers(
                self.documents,
                retrievers=val_retrievers,
                scorer=duobert_scorer,
                device=self.device,
            ),
            validation_interval=cfg.monobert.validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG": False},
        )

        # The learner for the duobert.
        learner = Learner(
            # Misc settings
            device=self.device,
            random=self.random,
            # How to train the model
            trainer=duobert_trainer,
            # The model to train
            scorer=duobert_scorer,
            # Optimization settings
            steps_per_epoch=self.duobert.steps_per_epoch,
            optimizers=get_optimizers(self.optimizers),
            max_epochs=self.duobert.max_epochs,
            # The listeners (here, for validation)
            listeners={"bestval": duobert_validation},
            # The hook used for evaluation
            hooks=[setmeta(DistributedHook(models=[duobert_scorer]), True)],
        )

        # Submit job and link
        outputs = learner.submit(launcher=self.launcher_learner)

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
                model_id=f"duobert-{metric_name}",
            )

        return PaperResults(
            models={"duobert-RR@10": outputs.listeners["bestval"]["RR@10"]},
            evaluations=self.tests,
            tb_logs={"duobert-RR@10": learner.logpath},
        )


@paper_command(package=__package__)
def cli(xp: experiment, cfg: DuoBERT):
    return DuoBERTExperiment(xp, cfg).run()
