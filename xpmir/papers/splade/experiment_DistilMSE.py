# Implementation of the experiments in the paper SPLADE v2: Sparse Lexical and
# Expansion Model for Information Retrieval, (Thibault Formal, Carlos Lassance,
# Benjamin Piwowarski, Stéphane Clinchant), 2021
# https://arxiv.org/abs/2109.10086

import logging
from pathlib import Path
import os
from typing import Dict, List

from omegaconf import OmegaConf
from datamaestro import prepare_dataset
from experimaestro.launcherfinder import find_launcher

from experimaestro import experiment, tag, tagspath, copyconfig, setmeta
from experimaestro.click import click
from experimaestro.utils import cleanupdir
from xpmir.configuration import omegaconf_argument
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
from xpmir.letor.optim import AdamW, RegexParameterFilter, get_optimizers
from xpmir.letor.schedulers import LinearWithWarmup
from xpmir.index.faiss import IndexBackedFaiss, FaissRetriever
from xpmir.index.sparse import (
    SparseRetriever,
    SparseRetrieverIndexBuilder,
)
from xpmir.rankers import Scorer
from xpmir.rankers.full import FullRetriever
from xpmir.letor.trainers import Trainer
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.neural.dual import DenseDocumentEncoder, DenseQueryEncoder
from xpmir.letor.optim import ParameterOptimizer
from xpmir.rankers.standard import BM25
from xpmir.neural.splade import spladeV2_max
from xpmir.measures import AP, P, nDCG, RR
from xpmir.neural.pretrained import tas_balanced

logging.basicConfig(level=logging.INFO)


@click.option("--debug", is_flag=True, help="Print debug information")
@click.option(
    "--env", help="Define one environment variable", type=(str, str), multiple=True
)
@click.option(
    "--host",
    type=str,
    default=None,
    help="Server hostname (default to localhost, not suitable if your jobs are remote)",
)
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@omegaconf_argument("configuration", package=__package__)
@click.argument("workdir", type=Path)
@click.command()
def cli(
    env: Dict[str, str],
    debug: bool,
    configuration,
    host: str,
    port: int,
    workdir: str,
    args,
):
    """Runs an experiment"""
    # Merge the additional option to the existing
    conf_args = OmegaConf.from_dotlist(args)
    configuration = OmegaConf.merge(configuration, conf_args)

    tags = configuration.Launcher.tags or []

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    # Number of topics in the validation set
    VAL_SIZE = configuration.Learner.validation_size

    # Number of steps in each epoch
    steps_per_epoch = configuration.Learner.steps_per_epoch

    # Validation interval (in epochs)
    validation_interval = configuration.Learner.validation_interval

    # How many documents retrieved from the base retriever(bm25)
    topK = configuration.base_retriever.topK

    # How many documents to be process once to in the FullRetrieverScorer(batch_size)
    batch_size_full_retriever = configuration.full_retriever.batch_size_full_retriever

    # the max epochs to train
    max_epochs = configuration.Learner.max_epochs

    # the batch_size for training the splade model
    splade_batch_size = configuration.Learner.splade_batch_size

    # The numbers of the warmup steps during the training
    num_warmup_steps = configuration.Learner.num_warmup_steps

    # # Top-K when building the validation set(tas-balanced)
    retTopK = configuration.tas_balance_retriever.retTopK

    # Validation interval (in epochs)
    validation_interval = configuration.Learner.validation_interval

    # After how many steps without improvement, the trainer stops.
    early_stop = configuration.Learner.early_stop

    # FAISS index building
    indexspec = configuration.tas_balance_retriever.indexspec
    faiss_max_traindocs = configuration.tas_balance_retriever.faiss_max_traindocs

    # the flop coefficient for query and documents
    lambda_q = configuration.Learner.lambda_q
    lambda_d = configuration.Learner.lambda_d
    lamdba_warmup_steps = configuration.Learner.lamdba_warmup_steps

    # the number of documents retrieved from the splade model during the evaluation
    topK_eval_splade = topK

    logging.info(
        f"Number of epochs {max_epochs}, validation interval {validation_interval}"
    )

    if (max_epochs % validation_interval) != 0:
        raise AssertionError(
            f"Number of epochs ({max_epochs}) is not "
            f"a multiple of validation interval ({validation_interval})"
        )

    name = configuration.type

    # launchers
    assert configuration.Launcher.gpu, "It is recommend to do this on GPU"
    cpu_launcher_index = find_launcher(configuration.Indexation.requirements)
    gpu_launcher_index = find_launcher(
        configuration.Indexation.training_requirements, tags=tags
    )
    gpu_launcher_learner = find_launcher(configuration.Learner.requirements, tags=tags)
    gpu_launcher_evaluate = find_launcher(
        configuration.Evaluation.requirements, tags=tags
    )

    # Starts the experiment
    with experiment(workdir, name, host=host, port=port) as xp:
        # Set environment variables
        xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])
        for key, value in env:
            xp.setenv(key, value)

        # Misc
        device = CudaDevice() if configuration.Launcher.gpu else Device()
        random = Random(seed=0)

        # prepare the dataset
        documents = prepare_dataset(
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

        # Not work for now, avoid shuffuling first.
        # TODO: Generalizing the shuffle mechanism for distillation

        # triplesid = ShuffledTrainingTripletsLines(
        #     seed=123,
        #     data=train_triples_distil,
        # ).submit()

        # Build a dev. collection for full-ranking (validation) "Efficiently
        # Teaching an Effective Dense Retriever with Balanced Topic Aware
        # Sampling"
        tasb = tas_balanced()  # create a scorer from huggingface

        # task to train the tas_balanced encoder for the document list and
        # generate an index for retrieval
        tasb_index = IndexBackedFaiss(
            indexspec=indexspec,
            device=device,
            normalize=False,
            documents=documents,
            sampler=RandomDocumentSampler(
                documents=documents, max_count=faiss_max_traindocs
            ),  # Just use a fraction of the dataset for training
            encoder=DenseDocumentEncoder(scorer=tasb),
            batchsize=2048,
            batcher=PowerAdaptativeBatcher(),
            hooks=[
                setmeta(
                    DistributedHook(models=[tasb.encoder, tasb.query_encoder]), True
                )
            ],
        ).submit(launcher=gpu_launcher_index)

        # A retriever if tas-balanced. We use the index of the faiss.
        # Used it to create the validation dataset.
        tasb_retriever = FaissRetriever(
            index=tasb_index, topk=retTopK, encoder=DenseQueryEncoder(scorer=tasb)
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
                dataset=dev, seed=123, fold=0, sizes=[VAL_SIZE], exclude=devsmall.topics
            ).submit(),
            retrievers=[
                tasb_retriever,
                AnseriniRetriever(k=retTopK, index=index, model=basemodel),
            ],
        ).submit(launcher=gpu_launcher_index)

        # compute the baseline performance on the test dataset.
        # Bm25
        bm25_retriever = AnseriniRetriever(k=topK, index=index, model=basemodel)
        tests.evaluate_retriever(
            copyconfig(bm25_retriever).tag("model", "bm25"), cpu_launcher_index
        )

        # tas-balance
        tests.evaluate_retriever(
            copyconfig(tasb_retriever).tag("model", "tasb"), gpu_launcher_index
        )

        # define the path to store the result for tensorboard
        runspath = xp.resultspath / "runs"
        cleanupdir(runspath)
        runspath.mkdir(exist_ok=True, parents=True)

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

        # scheduler for trainer
        scheduler = LinearWithWarmup(num_warmup_steps=num_warmup_steps)

        # Define the model and the flop loss for regularization
        # Model of class: DotDense()
        # The parameters are the regularization coeff for the query and document
        spladev2, flops = spladeV2_max(lambda_q, lambda_d, lamdba_warmup_steps)

        # Base retrievers for validation
        # It retrieve all the document of the collection with score 0
        base_retriever_full = FullRetriever(documents=ds_val.documents)

        distil_pairwise_trainer = DistillationPairwiseTrainer(
            batch_size=splade_batch_size,
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
                    batch_size_full_retriever,
                    PowerAdaptativeBatcher(),
                    device=device,
                ),
                early_stop=early_stop,
                validation_interval=validation_interval,
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
                steps_per_epoch=steps_per_epoch,
                use_fp16=True,
                max_epochs=tag(max_epochs),
                # the listener for the validation
                listeners={"bestval": validation},
                # the hooks
                hooks=hooks,
            )
            # submit the learner and build the symbolique link
            outputs = learner.submit(launcher=launcher)
            (runspath / tagspath(learner)).symlink_to(learner.logpath)

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
                topk=topK_eval_splade,
                batchsize=1,
                encoder=DenseQueryEncoder(scorer=scorer),
            )

        # Do the training process and then return the best model for splade
        best_model = run(
            spladev2.tag("model", "splade-v2"),
            distil_pairwise_trainer,
            [
                ParameterOptimizer(
                    scheduler=scheduler,
                    optimizer=AdamW(lr=configuration.Learner.lr),
                    filter=RegexParameterFilter(
                        includes=[r"\.bias$", r"\.LayerNorm\."]
                    ),
                ),
                ParameterOptimizer(
                    scheduler=scheduler,
                    optimizer=AdamW(lr=configuration.Learner.lr),
                    filter=RegexParameterFilter(
                        excludes=[r"\.bias$", r"\.LayerNorm\."]
                    ),
                ),
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
        tests.evaluate_retriever(splade_retriever, gpu_launcher_evaluate)

        # wait for all the experiments ends
        xp.wait()

        # Display metrics for each trained model
        tests.output_results()


if __name__ == "__main__":
    cli()
