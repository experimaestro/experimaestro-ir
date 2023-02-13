# Implementation of the experiments in the paper SPLADE v2: Sparse Lexical and
# Expansion Model for Information Retrieval, (Thibault Formal, Carlos Lassance,
# Benjamin Piwowarski, Stéphane Clinchant), 2021
# https://arxiv.org/abs/2109.10086

import logging
import os
from typing import List

from datamaestro_text.data.ir import AdhocDocuments
from datamaestro import prepare_dataset
from experimaestro.launcherfinder import find_launcher

from experimaestro import experiment, tag, copyconfig, setmeta
from xpmir.datasets.adapters import (
    MemoryTopicStore,
    RandomFold,
    RetrieverBasedCollection,
)
from xpmir.distributed import DistributedHook
from xpmir.documents.samplers import RandomDocumentSampler
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor.devices import CudaDevice, Device
from xpmir.letor import Random
from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)
from xpmir.letor.distillation.samplers import (
    DistillationPairwiseSampler,
    PairwiseHydrator,
)
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.schedulers import LinearWithWarmup
from xpmir.index.faiss import IndexBackedFaiss, FaissRetriever
from xpmir.index.sparse import (
    SparseRetriever,
    SparseRetrieverIndexBuilder,
)
from xpmir.models import AutoModel
from xpmir.papers.cli import paper_command, UploadToHub
from xpmir.rankers import Scorer
from xpmir.rankers.full import FullRetriever
from xpmir.letor.trainers import Trainer
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.neural.dual import DenseDocumentEncoder, DenseQueryEncoder
from xpmir.letor.optim import (
    AdamW,
    ParameterOptimizer,
    get_optimizers,
    TensorboardService,
)
from xpmir.rankers.standard import BM25
from xpmir.neural.splade import spladeV2_max
from xpmir.measures import AP, P, nDCG, RR
from .configuration import SPLADE

logging.basicConfig(level=logging.INFO)
# could be deleted, not not sure yet.


@paper_command(schema=SPLADE, package=__package__)
def cli(xp: experiment, cfg: SPLADE, upload_to_hub: UploadToHub):
    """SPLADE_DistilMSE: SPLADEv2 trained with the distillated triplets

    Training data from: https://github.com/sebastian-hofstaetter/neural-ranking-kd

    From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models
    More Effective (Thibault Formal, Carlos Lassance, Benjamin Piwowarski,
    Stéphane Clinchant). 2022. https://arxiv.org/abs/2205.04733
    """

    # Defining the different launchers
    cpu_launcher_index = find_launcher(cfg.indexation.requirements)
    gpu_launcher_index = find_launcher(cfg.indexation.training_requirements)
    gpu_launcher_learner = find_launcher(cfg.learner.requirements)
    gpu_launcher_evaluate = find_launcher(cfg.evaluation.requirements)

    # Sets the working directory and the name of the xp
    # Needed by Pyserini
    xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])

    # Misc
    device = CudaDevice() if cfg.gpu else Device()
    random = Random(seed=0)

    # prepare the dataset
    documents: AdhocDocuments = prepare_dataset(
        "irds.msmarco-passage.documents"
    )  # all the documents for msmarco

    # All the query text
    train_topics = prepare_dataset("irds.msmarco-passage.train.queries")

    dev = prepare_dataset("irds.msmarco-passage.dev")

    # For evaluation
    devsmall = prepare_dataset("irds.msmarco-passage.dev.small")  # development

    # hard negatives trained by distillation with cross-encoder
    # Improving Efficient Neural Ranking Models with Cross-Architecture
    # Knowledge Distillation, (Sebastian Hofstätter, Sophia Althammer,
    # Michael Schröder, Mete Sertkan, Allan Hanbury), 2020
    # In the form of Tuple[Query, Tuple[Document, Document]] without text
    train_triples_distil = prepare_dataset(
        "com.github.sebastian-hofstaetter."
        + "neural-ranking-kd.msmarco.ensemble.teacher"
    )

    # Index for msmarcos
    index = IndexCollection(documents=documents, storeContents=True).submit()

    # Build a dev. collection for full-ranking (validation) "Efficiently
    # Teaching an Effective Dense Retriever with Balanced Topic Aware
    # Sampling"
    tasb = AutoModel.load_from_hf_hub(
        "xpmir/tas-balanced"
    )  # create a scorer from huggingface

    # task to train the tas_balanced encoder for the document list and
    # generate an index for retrieval
    tasb_index = IndexBackedFaiss(
        indexspec=cfg.tas_balance_retriever.indexspec,
        device=device,
        normalize=False,
        documents=documents,
        sampler=RandomDocumentSampler(
            documents=documents, max_count=cfg.tas_balance_retriever.faiss_max_traindocs
        ),  # Just use a fraction of the dataset for training
        encoder=DenseDocumentEncoder(scorer=tasb),
        batchsize=2048,
        batcher=PowerAdaptativeBatcher(),
        hooks=[
            setmeta(DistributedHook(models=[tasb.encoder, tasb.query_encoder]), True)
        ],
    ).submit(launcher=gpu_launcher_index)

    # A retriever if tas-balanced. We use the index of the faiss.
    # Used it to create the validation dataset.
    tasb_retriever = FaissRetriever(
        index=tasb_index,
        topk=cfg.tas_balance_retriever.retTopK,
        encoder=DenseQueryEncoder(scorer=tasb),
    )

    # also the bm25 for creating the validation set.
    basemodel = BM25()

    # define the evaluation measures and dataset.
    measures = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10]
    tests: EvaluationsCollection = EvaluationsCollection(
        trec2019=Evaluations(
            prepare_dataset("irds.msmarco-passage.trec-dl-2019"), measures
        ),
        msmarco_dev=Evaluations(devsmall, measures),
    )

    # building the validation dataset. Based on the existing dataset and the
    # top retrieved doc from tas-balanced and bm25
    ds_val = RetrieverBasedCollection(
        dataset=RandomFold(
            dataset=dev,
            seed=123,
            fold=0,
            sizes=[cfg.learner.validation_size],
            exclude=devsmall.topics,
        ).submit(),
        retrievers=[
            tasb_retriever,
            AnseriniRetriever(
                k=cfg.tas_balance_retriever.retTopK, index=index, model=basemodel
            ),
        ],
    ).submit(launcher=gpu_launcher_index)

    # compute the baseline performance on the test dataset.
    # Bm25
    bm25_retriever = AnseriniRetriever(
        k=cfg.base_retriever.topK, index=index, model=basemodel
    )
    tests.evaluate_retriever(
        copyconfig(bm25_retriever).tag("model", "bm25"), cpu_launcher_index
    )

    # tas-balance
    tests.evaluate_retriever(
        copyconfig(tasb_retriever).tag("model", "tasb"), gpu_launcher_index
    )

    # define the path to store the result for tensorboard
    tb = xp.add_service(TensorboardService(xp.resultspath / "runs"))

    # Combine the training triplets with the document and queries texts
    distillation_samples = PairwiseHydrator(
        samples=train_triples_distil,
        documentstore=documents,
        querystore=MemoryTopicStore(topics=train_topics),
    )

    # Generate a sampler from the samples
    distil_pairwise_sampler = DistillationPairwiseSampler(samples=distillation_samples)

    # scheduler for trainer
    scheduler = LinearWithWarmup(num_warmup_steps=cfg.learner.num_warmup_steps)

    # Define the model and the flop loss for regularization
    # Model of class: DotDense()
    # The parameters are the regularization coeff for the query and document
    spladev2, flops = spladeV2_max(
        cfg.learner.lambda_q, cfg.learner.lambda_d, cfg.learner.lamdba_warmup_steps
    )

    # Base retrievers for validation
    # It retrieve all the document of the collection with score 0
    base_retriever_full = FullRetriever(documents=ds_val.documents)

    distil_pairwise_trainer = DistillationPairwiseTrainer(
        batch_size=cfg.learner.splade_batch_size,
        sampler=distil_pairwise_sampler,
        lossfn=MSEDifferenceLoss(),
        hooks=[flops],
    )

    # run the learner and do the evaluation with the best result
    def run(
        scorer: Scorer,
        trainer: Trainer,
        optimizers: List,
        hooks=[],
        launcher=gpu_launcher_learner,
    ):
        # establish the validation listener
        validation = ValidationListener(
            dataset=ds_val,
            # a retriever which use the splade model to score all the
            # documents and then do the retrieve
            retriever=scorer.getRetriever(
                base_retriever_full,
                cfg.full_retriever.batch_size_full_retriever,
                PowerAdaptativeBatcher(),
                device=device,
            ),
            early_stop=cfg.learner.early_stop,
            validation_interval=cfg.learner.validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG@10": False},
        )

        # the learner for the splade
        learner = Learner(
            # Misc settings
            random=random,
            device=device,
            # How to train the model
            trainer=trainer,
            # the model to be trained
            scorer=scorer,
            # Optimization settings
            optimizers=get_optimizers(optimizers),
            steps_per_epoch=cfg.learner.steps_per_epoch,
            use_fp16=True,
            max_epochs=tag(cfg.learner.max_epochs),
            # the listener for the validation
            listeners={"bestval": validation},
            # the hooks
            hooks=hooks,
        )
        # submit the learner and build the symbolique link
        outputs = learner.submit(launcher=launcher)
        tb.add(learner, learner.logpath)

        # return the best trained model here for only RR@10
        best = outputs.listeners["bestval"]["RR@10"]

        return best

    # Get a sparse retriever from a dual scorer
    def sparse_retriever(scorer, documents):
        """Builds a sparse retriever
        Used to evaluate the scorer
        """
        # build a retriever for the documents
        index = SparseRetrieverIndexBuilder(
            batch_size=512,
            batcher=PowerAdaptativeBatcher(),
            encoder=DenseDocumentEncoder(scorer=scorer),
            device=device,
            documents=documents,
            ordered_index=False,
        ).submit(launcher=gpu_launcher_index)

        return SparseRetriever(
            index=index,
            topk=cfg.base_retriever.topK,
            batchsize=1,
            encoder=DenseQueryEncoder(scorer=scorer),
        )

    # Do the training process and then return the best model for splade
    best_model = run(
        spladev2.tag("model", "splade-v2-distilMSE"),
        distil_pairwise_trainer,
        [
            ParameterOptimizer(
                scheduler=scheduler,
                optimizer=AdamW(lr=cfg.learner.lr),
            )
        ],
        hooks=[
            setmeta(
                DistributedHook(models=[spladev2.encoder, spladev2.query_encoder]),
                True,
            )
        ],
        launcher=gpu_launcher_learner,
    )

    # evaluate the best model
    splade_retriever = sparse_retriever(best_model, documents)

    tests.evaluate_retriever(
        splade_retriever, gpu_launcher_evaluate, model_id="SPLADE_DistilMSE-RR@10"
    )

    # wait for all the experiments ends
    xp.wait()

    # Upload to HUB if requested
    upload_to_hub.send_scorer({"SPLADE_DistilMSE-RR@10": best_model}, evaluations=tests)

    # Display metrics for each trained model
    tests.output_results()


if __name__ == "__main__":
    cli()
