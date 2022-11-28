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
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions
from experimaestro.utils import cleanupdir
from experimaestro.launcherfinder.specs import duration
from xpmir.configuration import omegaconf_argument
from xpmir.datasets.adapters import RandomFold, RetrieverBasedCollection
from xpmir.documents.samplers import RandomDocumentSampler
from xpmir.evaluation import Evaluate
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor.devices import CudaDevice, Device
from xpmir.letor import Random
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.optim import Adam, AdamW
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler, TripletBasedSampler
from xpmir.letor.schedulers import CosineWithWarmup
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
@click.option("--tags", type=str, default="", help="Tags for selecting the launcher")
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
    tags: str,
    env: Dict[str, str],
    debug: bool,
    gpu: bool,
    configuration, 
    host: str,
    port: int,
    workdir: str
):
    """Runs an experiment"""
    tags = tags.split(",") if tags else []
    
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    if max_epochs is not None:
        max_epochs = int(max_epochs)
    
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
    
    # num_warmup_steps = 1000

    # # Top-K when building the validation set(tas-balanced)
    # retTopK = 10

    # # Validation interval (in epochs)
    # validation_interval = max_epochs // 4

    # early_stop = 0

    # # FAISS index building
    # indexspec = "OPQ4_16,IVF256_HNSW32,PQ4"
    # faiss_max_traindocs = 15_000



if __name__ == "__main__":
    cli()