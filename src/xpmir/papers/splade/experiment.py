# Implementation of the experiments in the paper SPLADE v2: Sparse Lexical and
# Expansion Model for Information Retrieval, (Thibault Formal, Carlos Lassance,
# Benjamin Piwowarski, Stéphane Clinchant), 2021
# https://arxiv.org/abs/2109.10086

import logging
from functools import lru_cache

from experimaestro.launcherfinder import find_launcher

from experimaestro import experiment, tag, setmeta
from xpmir.distributed import DistributedHook
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.schedulers import LinearWithWarmup
from xpmir.index.sparse import (
    SparseRetriever,
    SparseRetrieverIndexBuilder,
)
from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)
from xpmir.papers.cli import paper_command
from xpmir.letor.trainers.batchwise import BatchwiseTrainer, SoftmaxCrossEntropy
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.neural.dual import DenseDocumentEncoder, DenseQueryEncoder
from xpmir.letor.optim import (
    ParameterOptimizer,
    AdamW,
    get_optimizers,
)
from xpmir.rankers.standard import BM25
from xpmir.neural.splade import spladeV2_max, spladeV2_doc
from xpmir.papers.results import PaperResults
from .pipelines import SPLADEMSMarcoV1Experiment
from .configuration import SPLADE, Learner as LearnerConfig

logging.basicConfig(level=logging.INFO)

# Run by:
# $ xpmir papers splade spladeV2 --configuration config_name experiment/


class SPLADEExperiment(SPLADEMSMarcoV1Experiment):
    """SPLADEv2 models"""

    cfg: SPLADE

    basemodel = BM25()

    def __init__(self, xp: experiment, cfg: SPLADE):
        super().__init__(xp, cfg)
        self.gpu_launcher_learner = find_launcher(cfg.learner.requirements)
        self.gpu_launcher_evaluate = find_launcher(cfg.evaluation.requirements)

    @lru_cache
    def get_optimizers(self, cfg: LearnerConfig):
        scheduler = (
            LinearWithWarmup(num_warmup_steps=cfg.num_warmup_steps)
            if cfg.scheduler
            else None
        )

        return get_optimizers(
            [
                ParameterOptimizer(
                    scheduler=scheduler,
                    optimizer=AdamW(lr=cfg.lr),
                )
            ]
        )

    def run(self) -> PaperResults:
        """SPLADE model"""

        cfg = self.cfg
        # -----Learning to rank component preparation part-----

        # Define the model and the flop loss for regularization
        # Model of class: DotDense()
        # The parameters are the regularization coeff for the query and document
        if cfg.learner.model == "splade_max":
            spladev2, flops = spladeV2_max(
                cfg.learner.lambda_q,
                cfg.learner.lambda_d,
                cfg.learner.lamdba_warmup_steps,
            )
        elif cfg.learner.model == "splade_doc":
            spladev2, flops = spladeV2_doc(
                cfg.learner.lambda_q,
                cfg.learner.lambda_d,
                cfg.learner.lamdba_warmup_steps,
            )
        else:
            raise NotImplementedError

        # define the trainer based on different dataset
        if cfg.learner.dataset == "":
            batchwise_trainer_flops = BatchwiseTrainer(
                batch_size=cfg.learner.splade_batch_size,
                sampler=self.splade_sampler,
                lossfn=SoftmaxCrossEntropy(),
                hooks=[flops],
            )
        elif cfg.learner.dataset == "bert_hard_negative":
            batchwise_trainer_flops = DistillationPairwiseTrainer(
                batch_size=cfg.learner.splade_batch_size,
                sampler=self.splade_sampler,
                lossfn=MSEDifferenceLoss(),
                hooks=[flops],
            )

        # hooks for the learner
        if cfg.learner.model == "splade_doc":
            hooks = [
                setmeta(
                    DistributedHook(models=[spladev2.encoder]),
                    True,
                )
            ]
        else:
            hooks = [
                setmeta(
                    DistributedHook(models=[spladev2.encoder, spladev2.query_encoder]),
                    True,
                )
            ]

        # establish the validation listener
        validation = ValidationListener(
            dataset=self.ds_val,
            # a retriever which use the splade model to score all the
            # documents and then do the retrieve
            retriever=spladev2.getRetriever(
                self.base_retriever_full,
                cfg.full_retriever.batch_size_full_retriever,
                PowerAdaptativeBatcher(),
                device=self.device,
            ),
            early_stop=cfg.learner.early_stop,
            validation_interval=cfg.learner.validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG@10": False},
            store_last_checkpoint=True if cfg.learner.model == "splade_doc" else False,
        )

        # the learner: Put the components together
        learner = Learner(
            # Misc settings
            random=self.random,
            device=self.device,
            # How to train the model
            trainer=batchwise_trainer_flops,
            # the model to be trained
            scorer=spladev2.tag("model", "splade-v2"),
            # Optimization settings
            optimizers=self.get_optimizers(cfg.learner),
            steps_per_epoch=cfg.learner.steps_per_epoch,
            use_fp16=True,
            max_epochs=tag(cfg.learner.max_epochs),
            # the listener for the validation
            listeners={"bestval": validation},
            # the hooks
            hooks=hooks,
        )

        # submit the learner and build the symbolique link
        outputs = learner.submit(launcher=self.gpu_launcher_learner)
        self.tb.add(learner, learner.logpath)

        # get the trained model
        trained_model = (
            outputs.listeners["bestval"]["last_checkpoint"]
            if cfg.learner.model == "splade_doc"
            else outputs.listeners["bestval"]["RR@10"]
        )

        # build a retriever for the documents
        sparse_index = SparseRetrieverIndexBuilder(
            batch_size=512,
            batcher=PowerAdaptativeBatcher(),
            encoder=DenseDocumentEncoder(scorer=trained_model),
            device=self.device,
            documents=self.documents,
            ordered_index=False,
        ).submit(launcher=self.gpu_launcher_index)

        # Build the sparse retriever based on the index
        splade_retriever = SparseRetriever(
            index=sparse_index,
            topk=cfg.base_retriever.topK,
            batchsize=1,
            encoder=DenseQueryEncoder(scorer=trained_model),
        )

        # evaluate the best model
        self.tests.evaluate_retriever(
            splade_retriever,
            self.gpu_launcher_evaluate,
            model_id=f"{cfg.learner.model}-{cfg.learner.dataset}-RR@10",
        )

        return PaperResults(
            models={f"{cfg.learner.model}-{cfg.learner.dataset}-RR@10": trained_model},
            evaluations=self.tests,
            tb_logs={
                f"{cfg.learner.model}-{cfg.learner.dataset}-RR@10": learner.logpath
            },
        )


@paper_command(schema=SPLADE, package=__package__)
def cli(xp: experiment, cfg: SPLADE):
    return SPLADEExperiment(xp, cfg).run()
