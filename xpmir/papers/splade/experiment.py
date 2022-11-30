# Implementation of the experiments in the paper
# SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval, (Thibault Formal, Carlos Lassance, Benjamin Piwowarski, Stéphane Clinchant), 2021
# https://arxiv.org/abs/2109.10086

import logging
from pathlib import Path
import os
from typing import Dict, List, Optional


from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from experimaestro.launcherfinder import cpu, cuda_gpu, find_launcher

from experimaestro import experiment, tag, tagspath, copyconfig, setmeta
from experimaestro.click import click, forwardoption
from experimaestro.utils import cleanupdir
from experimaestro.launcherfinder.specs import duration
from xpmir.configuration import omegaconf_argument
from xpmir.datasets.adapters import RandomFold, RetrieverBasedCollection
from xpmir.distributed import DistributedHook
from xpmir.documents.samplers import RandomDocumentSampler
from xpmir.evaluation import Evaluate, Evaluations, EvaluationsCollection
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor.devices import CudaDevice, Device
from xpmir.letor import Random
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.optim import Adam, AdamW, get_optimizers
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler, TripletBasedSampler
from xpmir.letor.schedulers import CosineWithWarmup, LinearWithWarmup
from xpmir.index.faiss import IndexBackedFaiss, FaissRetriever
from xpmir.index.sparse import (
    SparseRetriever,
    SparseRetrieverIndexBuilder,
)
from xpmir.rankers import RandomScorer, Scorer
from xpmir.rankers.full import FullRetriever
from xpmir.letor.trainers import Trainer, pointwise
from xpmir.letor.trainers.batchwise import BatchwiseTrainer, SoftmaxCrossEntropy
from xpmir.letor.batchers import PowerAdaptativeBatcher
import xpmir.letor.trainers.pairwise as pairwise
from xpmir.neural.dual import Dense, DenseDocumentEncoder, DenseQueryEncoder, DotDense
from xpmir.letor.optim import ParameterOptimizer
from xpmir.rankers.standard import BM25
from xpmir.neural.splade import DistributedSpladeTextEncoderHook, spladeV2
from xpmir.measures import AP, P, nDCG, RR
from xpmir.neural.pretrained import tas_balanced
from xpmir.text.huggingface import DistributedModelHook, TransformerEncoder

logging.basicConfig(level=logging.INFO)

# @forwardoption.max_epochs(Learner, default=None)
# @click.option("--tags", type=str, default="", help="Tags for selecting the launcher")
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option(
    "--env", help="Define one environment variable", type=(str, str), multiple=True
)
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option(
    "--batch-size", type=int, default=None, help="Batch size (validation and test)"
)
@click.option(
    "--host",
    type=str,
    default=None,
    help="Server hostname (default to localhost, not suitable if your jobs are remote)",
)
@click.option("--port", type=int, default=12345, help="Port for monitoring")
# @click.option(
#     "--batch-size", type=int, default=None, help="Batch size (validation and test)"
# )

# @omegaconf_argument("configuration", package=__package__)
# works only with this one a the moment
@omegaconf_argument("configuration", package='xpmir.papers.splade')
@click.argument("workdir", type=Path)
@click.command()

def cli(
    env: Dict[str, str],
    debug: bool,
    gpu: bool,
    configuration, 
    host: str,
    port: int,
    workdir: str
):
    """Runs an experiment"""
    tags = configuration.Launcher.tags.split(",") if configuration.Launcher.tags else []
    
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
    early_stop = 0

    # FAISS index building
    indexspec = "OPQ4_16,IVF256_HNSW32,PQ4"
    faiss_max_traindocs = 15_000

    # the number of documents retrieved from the splade model during the evaluation
    topK_eval_splade = topK

    logging.info(
        f"Number of epochs {max_epochs}, validation interval {validation_interval}"
    )

    if (max_epochs % validation_interval) != 0:
        raise AssertionError(
            f"Number of epochs ({max_epochs}) is not a multiple of validation interval ({validation_interval})"
        )

    name = configuration.type

    # launchers 
    req_duration = duration("6 days")

    cpu_launcher_4G = find_launcher(cuda_gpu(mem="4G"))
    launcher_basic = find_launcher(cpu() & req_duration, tags=tags)

    gpu_launcher_index = find_launcher((cuda_gpu(mem='24G')) & req_duration, tags=tags)
    gpu_launcher_learner = find_launcher((cuda_gpu(mem='48G')) & req_duration, tags=tags)

    # Starts the experiment
    with experiment(workdir, name, host=host, port=port, launcher=launcher_basic) as xp:
         # Set environment variables
        xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])
        for key, value in env:
            xp.setenv(key, value)

        # Misc
        device = CudaDevice() if configuration.Launcher.gpu else Device()
        random = Random(seed=0)

        # prepare the dataset 
        documents = prepare_dataset("irds.msmarco-passage.documents") # all the documents for msmarco
        dev = prepare_dataset("irds.msmarco-passage.dev")
        devsmall = prepare_dataset("irds.msmarco-passage.dev.small") # development
        train_triples = prepare_dataset("irds.msmarco-passage.train.docpairs") # pair for pairwise learner

        # Index for msmarcos
        index = IndexCollection(documents=documents, storeContents=True).submit()

        triplesid = ShuffledTrainingTripletsLines(
            seed=123,
            data=train_triples,
        ).submit()

        # Build a dev. collection for full-ranking (validation)
        # "Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling"
        tasb = tas_balanced() # create a scorer from huggingface

        # task to train the tas_balanced encoder for the document list and generate an index for retrieval
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
            # maybe use the generalized version for the hooks --> DistributedHook
            hooks=[
                setmeta(
                    DistributedModelHook(transformer=tasb.query_encoder.encoder), True
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
            trec2020=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2020"), measures
            ),
            msmarco_dev=Evaluations(devsmall, measures)
        )

        # building the validation dataset.
        # Based on the existing dataset and the top retrieved doc from tas-balanced and bm25
        ds_val = RetrieverBasedCollection(
            dataset=RandomFold(
                dataset=dev, seed=123, fold=0, sizes=[VAL_SIZE], exclude=devsmall.topics
            ).submit(),
            retrievers=[
                tasb_retriever,
                AnseriniRetriever(k=retTopK, index=index, model=basemodel),
            ],
        ).submit(launcher=gpu_launcher_learner)

        # compute the baseline performance on the test dataset.
        # Bm25 
        bm25_retriever = AnseriniRetriever(k=topK, index=index, model=basemodel)
        tests.evaluate_retriever(
            copyconfig(bm25_retriever).tag('model','bm25'), 
            cpu_launcher_4G
        )

        # tas-balance
        tests.evaluate_retriever(
            copyconfig(tasb_retriever).tag('model','bm25'),
            gpu_launcher_index
        )

        # define the path to store the result for tensorboard
        runspath = xp.resultspath / "runs"
        cleanupdir(runspath)
        runspath.mkdir(exist_ok=True, parents=True)

        # generator a batchwise sampler which is an Iterator of ProductRecords()
        train_sampler = TripletBasedSampler(source=triplesid, index=index) # the pairwise sampler from the dataset.
        ibn_sampler = PairwiseInBatchNegativesSampler(sampler=train_sampler) # generating the batchwise from the pairwise
        
        # scheduler for trainer
        scheduler = LinearWithWarmup(num_warmup_steps=num_warmup_steps)

        # Define the model and the flop loss for regularization
        # Model of class: DotDense()
        # The parameters are the regularization coeff for the query and document
        spladev2, flops = spladeV2(3e-4, 1e-4)

        # Base retrievers for validation
        # It retrieve all the document of the collection with score 0
        base_retriever_full = FullRetriever(documents=ds_val.documents)

        batchwise_trainer_flops = BatchwiseTrainer(
            batch_size=splade_batch_size,
            sampler=ibn_sampler,
            lossfn=SoftmaxCrossEntropy(),
            hooks=[flops],
        )

        # run the learner and do the evaluation with the best result
        def run(
            scorer: Scorer,
            trainer: Trainer,
            optimizers: List,
            create_retriever,
            hooks=[],
            launcher=gpu_launcher_learner
        ):
            # establish the validation listener
            validation = ValidationListener(
                dataset=ds_val,
                retriever=scorer.getRetriever(
                    base_retriever_full, batch_size_full_retriever, PowerAdaptativeBatcher(), device=device
                ),  # a retriever which use the splade model to score all the documents and then do the retrieve
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
            ...
            # to be continue tomorrow


        # Get a sparse retriever from a dual scorer
        def sparse_retriever(scorer, documents):
            """Builds a sparse retriever
            Used to evaluate the scorer
            """
            index = SparseRetrieverIndexBuilder(
                batch_size=512,
                batcher=PowerAdaptativeBatcher(),
                encoder=DenseDocumentEncoder(scorer=scorer),
                device=device,
                documents=documents,
            ).submit(launcher=gpu_launcher_index)

            return SparseRetriever(
                index=index,
                topk=topK,
                batchsize=256,
                encoder=DenseQueryEncoder(scorer=scorer),
            )

        run(
            spladev2.tag("model", "splade-v2"),
            batchwise_trainer_flops,
            [
                ParameterOptimizer(
                    scheduler=scheduler, optimizer=Adam(lr=configuration.Learner.lr)
                )
            ],
            lambda scorer: sparse_retriever(scorer, documents),
            hooks=[setmeta(DistributedHook(models=[spladev2.encoder]), True)],
            launcher=gpu_launcher_learner,
        )

        # wait for all the experiments ends
        xp.wait()

        # ---  End of the experiment
        # Display metrics for each trained model
        for key, dsevaluations in tests.collection.items():
            print(f"=== {key}")
            for evaluation in dsevaluations.results:
                print(
                    f"Results for {evaluation.__xpm__.tags()}\n{evaluation.results.read_text()}\n"
                )

if __name__ == "__main__":
    cli()