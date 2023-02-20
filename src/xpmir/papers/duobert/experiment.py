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
from xpmir.papers.cli import paper_command
from xpmir.text.huggingface import DualDuoBertTransformerEncoder
from xpmir.papers.monobert.experiment import MonoBERTExperiment
from xpmir.papers.results import PaperResults
from .configuration import DuoBERT

logging.basicConfig(level=logging.INFO)


class DuoBERTExperiment(MonoBERTExperiment):
    cfg: DuoBERT

    def run(self):
        cfg = self.cfg
        monobert_results = super().run()
        monobert_scorer = monobert_results.models["monobert-RR@10"]

        # ------Start the code for the duobert

        # Define the trainer for the duobert
        duobert_trainer = pairwise.DuoPairwiseTrainer(
            lossfn=pairwise.PairwiseLossWithTarget().tag("loss", "duo_logp"),
            sampler=self.train_sampler,
            batcher=PowerAdaptativeBatcher(),
            batch_size=cfg.duobert.batch_size,
        )

        # The scorer(model) for the duobert
        duobert_scorer: DuoCrossScorer = DuoCrossScorer(
            encoder=DualDuoBertTransformerEncoder(trainable=True, dropout=0.1)
        ).tag("duo-reranker", "duobert")

        # Validation: we use monoBERT but only keep validation_top_k
        # results

        monobert_val_retrievers = partial(
            self.model_based_retrievers,
            retrievers=partial(self.retrievers, k=cfg.duobert.base_validation_top_k),
            top_k=cfg.duobert.validation_top_k,
            scorer=monobert_scorer,
        )

        val_retriever = self.model_based_retrievers(
            self.documents, retrievers=monobert_val_retrievers, scorer=duobert_scorer
        )

        # The validation listener evaluates the full retriever
        # (retriever + reranker) and keep the best performing model
        # on the validation set
        validation = ValidationListener(
            dataset=self.ds_val,
            retriever=val_retriever,
            validation_interval=cfg.duobert.validation_interval,
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
            steps_per_epoch=cfg.duobert.steps_per_epoch,
            optimizers=self.get_optimizers(cfg.duobert),
            max_epochs=cfg.duobert.max_epochs,
            # The listeners (here, for validation)
            listeners={"bestval": validation},
            # The hook used for evaluation
            hooks=[setmeta(DistributedHook(models=[duobert_scorer]), True)],
            use_fp16=True,
        )

        # Submit job and link
        outputs = learner.submit(launcher=self.launcher_learner)
        self.tb.add(learner, learner.logpath)

        # Evaluate the neural model on test collections

        monobert_test_retrievers = partial(
            self.model_based_retrievers,
            retrievers=partial(self.retrievers, k=cfg.retrieval.base_k),
            top_k=cfg.retrieval.k,
            scorer=monobert_scorer,
        )
        test_retrievers = partial(
            self.model_based_retrievers,
            retrievers=monobert_test_retrievers,
            scorer=duobert_scorer,
        )

        for metric_name in validation.monitored():
            model = outputs.listeners["bestval"][metric_name]  # type: CrossScorer
            self.tests.evaluate_retriever(
                partial(
                    self.model_based_retrievers,
                    scorer=model,
                    retrievers=test_retrievers,
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


@paper_command(package=__package__, schema=DuoBERT)
def cli(xp: experiment, cfg: DuoBERT):
    return DuoBERTExperiment(xp, cfg).run()
