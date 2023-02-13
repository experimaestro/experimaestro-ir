# Implementation of the experiments in the paper SPLADE v2: Sparse Lexical and
# Expansion Model for Information Retrieval, (Thibault Formal, Carlos Lassance,
# Benjamin Piwowarski, Stéphane Clinchant), 2021
# https://arxiv.org/abs/2109.10086

import logging
import os
from typing import List

from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
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
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler, TripletBasedSampler
from xpmir.letor.schedulers import LinearWithWarmup
from xpmir.index.faiss import IndexBackedFaiss, FaissRetriever
from xpmir.index.sparse import (
    SparseRetriever,
    SparseRetrieverIndexBuilder,
)
from xpmir.letor.distillation.pairwise import (
    DistillationPairwiseTrainer,
    MSEDifferenceLoss,
)
from xpmir.letor.distillation.samplers import (
    DistillationPairwiseSampler,
    PairwiseHydrator,
)
from xpmir.models import AutoModel
from xpmir.papers.cli import paper_command, UploadToHub
from xpmir.rankers.full import FullRetriever
from xpmir.letor.trainers.batchwise import BatchwiseTrainer, SoftmaxCrossEntropy
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.neural.dual import DenseDocumentEncoder, DenseQueryEncoder
from xpmir.letor.optim import (
    ParameterOptimizer,
    AdamW,
    RegexParameterFilter,
    get_optimizers,
    TensorboardService,
)
from xpmir.rankers.standard import BM25
from xpmir.neural.splade import spladeV2_max, spladeV2_doc
from xpmir.measures import AP, P, nDCG, RR
from .configuration import SPLADE


logging.basicConfig(level=logging.INFO)


@paper_command(schema=SPLADE, package=__package__)
# Run by:
# $ xpmir papers splade spladeV2 --configuration config_name experiment/
def cli(xp: experiment, cfg: SPLADE, upload_to_hub: UploadToHub):
    """SPLADE: SPLADEv2 with max aggregation

    SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval
    (Thibault Formal, Carlos Lassance, Benjamin Piwowarski, Stéphane Clinchant).
    2021. https://arxiv.org/abs/2109.10086
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
    documents = prepare_dataset(
        "irds.msmarco-passage.documents"
    )  # all the documents for msmarco
    dev = prepare_dataset("irds.msmarco-passage.dev")
    devsmall = prepare_dataset("irds.msmarco-passage.dev.small")  # development
    # All the query text
    train_topics = prepare_dataset("irds.msmarco-passage.train.queries")

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

    # Base retrievers for validation
    # It retrieve all the document of the collection with score 0
    base_retriever_full = FullRetriever(documents=ds_val.documents)

    # define the path to store the result for tensorboard
    tb = xp.add_service(TensorboardService(xp.resultspath / "runs"))

    # -----Learning to rank component preparation part-----

    # Define the model and the flop loss for regularization
    # Model of class: DotDense()
    # The parameters are the regularization coeff for the query and document
    if cfg.learner.model == "splade_max":
        spladev2, flops = spladeV2_max(
            cfg.learner.lambda_q, cfg.learner.lambda_d, cfg.learner.lamdba_warmup_steps
        )
    elif cfg.learner.model == "splade_doc":
        spladev2, flops = spladeV2_doc(
            cfg.learner.lambda_q, cfg.learner.lambda_d, cfg.learner.lamdba_warmup_steps
        )
    else:
        raise NotImplementedError

    # define the trainer based on different dataset
    if cfg.learner.dataset == "":
        train_triples = prepare_dataset(
            "irds.msmarco-passage.train.docpairs"
        )  # pair for pairwise learner

        triplesid = ShuffledTrainingTripletsLines(
            seed=123,
            data=train_triples,
        ).submit()

        # generator a batchwise sampler which is an Iterator of ProductRecords()
        train_sampler = TripletBasedSampler(
            source=triplesid, index=index
        )  # the pairwise sampler from the dataset.
        ibn_sampler = PairwiseInBatchNegativesSampler(
            sampler=train_sampler
        )  # generating the batchwise from the pairwise
        batchwise_trainer_flops = BatchwiseTrainer(
            batch_size=cfg.learner.splade_batch_size,
            sampler=ibn_sampler,
            lossfn=SoftmaxCrossEntropy(),
            hooks=[flops],
        )
    elif cfg.learner.dataset == "bert_hard_negative":
        # hard negatives trained by distillation with cross-encoder
        # Improving Efficient Neural Ranking Models with Cross-Architecture
        # Knowledge Distillation, (Sebastian Hofstätter, Sophia Althammer,
        # Michael Schröder, Mete Sertkan, Allan Hanbury), 2020
        # In the form of Tuple[Query, Tuple[Document, Document]] without text
        train_triples_distil = prepare_dataset(
            "com.github.sebastian-hofstaetter."
            + "neural-ranking-kd.msmarco.ensemble.teacher"
        )
        # Combine the training triplets with the document and queries texts
        distillation_samples = PairwiseHydrator(
            samples=train_triples_distil,
            documentstore=documents,
            querystore=MemoryTopicStore(topics=train_topics),
        )

        # Generate a sampler from the samples
        distil_pairwise_sampler = DistillationPairwiseSampler(
            samples=distillation_samples
        )

        batchwise_trainer_flops = DistillationPairwiseTrainer(
            batch_size=cfg.learner.splade_batch_size,
            sampler=distil_pairwise_sampler,
            lossfn=MSEDifferenceLoss(),
            hooks=[flops],
        )

    # scheduler for trainer
    scheduler = LinearWithWarmup(num_warmup_steps=cfg.learner.num_warmup_steps)

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

    # FIXME: define the optimizer: due to a bug during the training of
    # splade_max and splade_doc
    distil_optimizers: List = [
        ParameterOptimizer(
            scheduler=scheduler,
            optimizer=AdamW(lr=cfg.learner.lr),
        )
    ]
    false_optimizers: List = [
        ParameterOptimizer(
            scheduler=scheduler,
            optimizer=AdamW(lr=cfg.learner.lr),
            filter=RegexParameterFilter(includes=[r"\.bias$", r"\.LayerNorm\."]),
        ),
        ParameterOptimizer(
            scheduler=scheduler,
            optimizer=AdamW(lr=cfg.learner.lr),
            filter=RegexParameterFilter(excludes=[r"\.bias$", r"\.LayerNorm\."]),
        ),
    ]

    # establish the validation listener
    validation = ValidationListener(
        dataset=ds_val,
        # a retriever which use the splade model to score all the
        # documents and then do the retrieve
        retriever=spladev2.getRetriever(
            base_retriever_full,
            cfg.full_retriever.batch_size_full_retriever,
            PowerAdaptativeBatcher(),
            device=device,
        ),
        early_stop=cfg.learner.early_stop,
        validation_interval=cfg.learner.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG@10": False},
        store_last_checkpoint=True if cfg.learner.model == "splade_doc" else False,
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
        optimizers=get_optimizers(
            false_optimizers if cfg.learner.dataset == "" else distil_optimizers
        ),
        steps_per_epoch=cfg.learner.steps_per_epoch,
        use_fp16=True,
        max_epochs=tag(cfg.learner.max_epochs),
        # the listener for the validation
        listeners={"bestval": validation},
        # the hooks
        hooks=hooks,
    )

    # submit the learner and build the symbolique link
    outputs = learner.submit(launcher=gpu_launcher_learner)
    tb.add(learner, learner.logpath)

    # get the trained model
    trained_model = (
        outputs.listeners["bestval"]["last_checkpoint"]
        if cfg.learner.model == "splade_doc"
        else outputs.listeners["bestval"]["RR@10"]
    )

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

    # Build a sparse index based on the trained model
    # and a retriever
    splade_retriever = sparse_retriever(trained_model, documents)

    # evaluate the best model
    tests.evaluate_retriever(
        splade_retriever,
        gpu_launcher_evaluate,
        model_id=f"{cfg.learner.model}-{cfg.learner.dataset}-RR@10",
    )

    # wait for all the experiments ends
    xp.wait()

    # Display metrics for each trained model
    tests.output_results()

    # Upload to HUB if requested
    upload_to_hub.send_scorer(
        {f"{cfg.learner.model}-{cfg.learner.dataset}-RR@10": trained_model},
        evaluations=tests,
    )


if __name__ == "__main__":
    cli()
