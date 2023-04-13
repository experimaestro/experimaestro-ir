# Implementation of the experiments in the paper SPLADE v2: Sparse Lexical and
# Expansion Model for Information Retrieval, (Thibault Formal, Carlos Lassance,
# Benjamin Piwowarski, Stéphane Clinchant), 2021
# https://arxiv.org/abs/2109.10086

import logging

from experimaestro.launcherfinder import find_launcher

from xpmir.letor.optim import (
    TensorboardService,
)
from experimaestro import experiment, setmeta
from xpmir.distributed import DistributedHook
from xpmir.letor.learner import Learner, ValidationListener
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
from xpmir.rankers.standard import BM25
from xpmir.neural.splade import spladeV2_max, spladeV2_doc
from xpmir.papers.results import PaperResults
from xpmir.papers.helpers.msmarco import (
    v1_tests,
    v1_validation_dataset,
    v1_passages,
)
from xpmir.datasets.adapters import RetrieverBasedCollection
from xpmir.papers.helpers.splade import splade_sampler
from xpmir.rankers.full import FullRetriever
from xpmir.documents.samplers import RandomDocumentSampler
from .configuration import SPLADE
import xpmir.interfaces.anserini as anserini
from xpmir.models import AutoModel
from xpmir.index.faiss import IndexBackedFaiss, FaissRetriever


logging.basicConfig(level=logging.INFO)

# Run by:
# $ xpmir papers splade spladeV2 --configuration config_name experiment/


def run(
    xp: experiment, cfg: SPLADE, tensorboard_service: TensorboardService
) -> PaperResults:
    """SPLADE model"""

    gpu_launcher_learner = find_launcher(cfg.splade.requirements)
    gpu_launcher_retrieval = find_launcher(cfg.retrieval.requirements)
    cpu_launcher_index = find_launcher(cfg.indexation.requirements)
    gpu_launcher_index = find_launcher(cfg.indexation.training_requirements)

    device = cfg.device
    random = cfg.random

    documents = v1_passages()
    ds_val_all = v1_validation_dataset(cfg.validation)

    tests = v1_tests()

    # -----The baseline------
    # BM25
    base_model = BM25().tag("model", "bm25")

    bm25_retriever = anserini.AnseriniRetriever(
        k=cfg.retrieval.topK, index=anserini.index_builder, model=base_model
    )

    tests.evaluate_retriever(bm25_retriever, cpu_launcher_index)

    # tas-balanced
    tasb = AutoModel.load_from_hf_hub("xpmir/tas-balanced").tag(
        "model", "tasb"
    )  # create a scorer from huggingface

    tasb_index = IndexBackedFaiss(
        indexspec=cfg.indexation.indexspec,
        device=device,
        normalize=False,
        documents=documents,
        sampler=RandomDocumentSampler(
            documents=documents,
            max_count=cfg.indexation.faiss_max_traindocs,
        ),  # Just use a fraction of the dataset for training
        encoder=DenseDocumentEncoder(scorer=tasb),
        batchsize=2048,
        batcher=PowerAdaptativeBatcher(),
        hooks=[
            setmeta(DistributedHook(models=[tasb.encoder, tasb.query_encoder]), True)
        ],
    ).submit(launcher=gpu_launcher_index)

    tasb_retriever = FaissRetriever(
        index=tasb_index,
        topk=cfg.retrieval.topK,
        encoder=DenseQueryEncoder(scorer=tasb),
    )

    tests.evaluate_retriever(tasb_retriever, gpu_launcher_index)

    # Building the validation set of the splade
    # We cannot use the full document dataset to build the validation set.

    # This one could be generic for both sparse and dense methods
    ds_val = RetrieverBasedCollection(
        dataset=ds_val_all,
        retrievers=[
            anserini.AnseriniRetriever(
                k=cfg.retrieval.retTopK, index=anserini.index_builder, model=base_model
            ),
        ],
    ).submit(launcher=gpu_launcher_index)

    # Base retrievers for validation
    # It retrieve all the document of the collection with score 0
    base_retriever_full = FullRetriever(documents=ds_val.documents)

    # -----Learning to rank component preparation part-----
    # Define the model and the flop loss for regularization
    # Model of class: DotDense()
    # The parameters are the regularization coeff for the query and document
    if cfg.splade.model == "splade_max":
        spladev2, flops = spladeV2_max(
            cfg.splade.lambda_q,
            cfg.splade.lambda_d,
            cfg.splade.lamdba_warmup_steps,
        )
    elif cfg.splade.model == "splade_doc":
        spladev2, flops = spladeV2_doc(
            cfg.splade.lambda_q,
            cfg.splade.lambda_d,
            cfg.splade.lamdba_warmup_steps,
        )
    else:
        raise NotImplementedError

    # define the trainer based on different dataset
    if cfg.splade.dataset == "":
        batchwise_trainer_flops = BatchwiseTrainer(
            batch_size=cfg.splade.optimization.batch_size,
            sampler=splade_sampler(),
            lossfn=SoftmaxCrossEntropy(),
            hooks=[flops],
        )
    elif cfg.splade.dataset == "bert_hard_negative":
        batchwise_trainer_flops = DistillationPairwiseTrainer(
            batch_size=cfg.splade.optimization.batch_size,
            sampler=splade_sampler(),
            lossfn=MSEDifferenceLoss(),
            hooks=[flops],
        )

    # hooks for the learner
    if cfg.splade.model == "splade_doc":
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
        id="bestval",
        dataset=ds_val,
        # a retriever which use the splade model to score all the
        # documents and then do the retrieve
        retriever=spladev2.getRetriever(
            base_retriever_full,
            cfg.retrieval.batch_size_full_retriever,
            PowerAdaptativeBatcher(),
            device=device,
        ),
        early_stop=cfg.splade.early_stop,
        validation_interval=cfg.splade.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG@10": False},
        store_last_checkpoint=True if cfg.splade.model == "splade_doc" else False,
    )

    # the learner: Put the components together
    learner = Learner(
        # Misc settings
        random=random,
        device=device,
        # How to train the model
        trainer=batchwise_trainer_flops,
        # the model to be trained
        scorer=spladev2.tag("model", "splade-v2"),
        # Optimization settings
        optimizers=cfg.splade.optimization.optimizer,
        steps_per_epoch=cfg.splade.optimization.steps_per_epoch,
        use_fp16=True,
        max_epochs=cfg.splade.optimization.max_epochs,
        # the listener for the validation
        listeners=[validation],
        # the hooks
        hooks=hooks,
    )

    # submit the learner and build the symbolique link
    outputs = learner.submit(launcher=gpu_launcher_learner)
    tensorboard_service.add(learner, learner.logpath)

    # get the trained model
    trained_model = (
        outputs.listeners["bestval"]["last_checkpoint"]
        if cfg.splade.model == "splade_doc"
        else outputs.listeners["bestval"]["RR@10"]
    )

    # build a retriever for the documents
    sparse_index = SparseRetrieverIndexBuilder(
        batch_size=512,
        batcher=PowerAdaptativeBatcher(),
        encoder=DenseDocumentEncoder(scorer=trained_model),
        device=device,
        documents=documents,
        ordered_index=False,
    ).submit(launcher=gpu_launcher_index)

    # Build the sparse retriever based on the index
    splade_retriever = SparseRetriever(
        index=sparse_index,
        topk=cfg.retrieval.topK,
        batchsize=1,
        encoder=DenseQueryEncoder(scorer=trained_model),
    )

    # evaluate the best model
    tests.evaluate_retriever(
        splade_retriever,
        gpu_launcher_retrieval,
        model_id=f"{cfg.splade.model}-{cfg.splade.dataset}-RR@10",
    )

    return PaperResults(
        models={f"{cfg.splade.model}-{cfg.splade.dataset}-RR@10": trained_model},
        evaluations=tests,
        tb_logs={f"{cfg.splade.model}-{cfg.splade.dataset}-RR@10": learner.logpath},
    )


@paper_command(schema=SPLADE, package=__package__)
def cli(xp: experiment, cfg: SPLADE, tensorboard_service: TensorboardService):
    return run(xp, cfg, tensorboard_service)
