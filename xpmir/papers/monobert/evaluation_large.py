# Implementation of the experiments in the paper
# Passage Re-ranking with BERT, (Rodrigo Nogueira, Kyunghyun Cho). 2019
# https://arxiv.org/abs/1901.04085

import dataclasses
import logging
from pathlib import Path
from omegaconf import OmegaConf
from xpmir.distributed import DistributedHook
from xpmir.letor.schedulers import LinearWithWarmup

import xpmir.letor.trainers.pairwise as pairwise
from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from xpmir.neural.cross import CrossScorer, DuoCrossScorer
from experimaestro import experiment, setmeta
from experimaestro.click import click, forwardoption
from experimaestro.launcherfinder import cpu, cuda_gpu, find_launcher
from experimaestro.launcherfinder.specs import duration
from experimaestro.utils import cleanupdir
from xpmir.configuration import omegaconf_argument
from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor import Device, Random
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.letor.devices import CudaDevice
from xpmir.letor.learner import Learner
from xpmir.letor.optim import Adam, ParameterOptimizer
from xpmir.letor.samplers import TripletBasedSampler
from xpmir.measures import AP, RR, P, nDCG
from xpmir.neural.jointclassifier import JointClassifier
from xpmir.pipelines.reranking import RerankingPipeline
from xpmir.rankers import RandomScorer, Retriever
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import DualTransformerEncoder, CrossScorerHuggingface
from xpmir.utils import find_java_home

logging.basicConfig(level=logging.INFO)

# --- Experiment
@forwardoption.max_epochs(Learner, default=None)
@click.option("--tags", type=str, default="", help="Tags for selecting the launcher")
@click.option("--debug", is_flag=True, help="Print debug information")
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
@click.option(
    "--port", type=int, default=12345, help="Port for monitoring (default 12345)"
)

# @omegaconf_argument("configuration", package=__package__)
# works only with this one a the moment
@omegaconf_argument("configuration", package="xpmir.papers.monobert")
@click.argument("workdir", type=Path)
@click.command()
def cli(debug, configuration, gpu, tags, host, port, workdir, max_epochs, batch_size):
    """Runs an experiment"""
    tags = tags.split(",") if tags else []

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    # How many document to re-rank for the monobert
    topK = configuration.Retriever_mono.k

    # Our default launcher for light tasks
    req_duration = duration("6 days")
    launcher = find_launcher(cpu() & req_duration, tags=tags)

    batch_size = batch_size or configuration.Learner.batch_size

    # FIXME: uses CPU constraints rather than GPU
    cpu_launcher_4g = find_launcher(cuda_gpu(mem="1G"))

    if configuration.type == "small":
        # We request a GPU, and if none, a CPU
        gpu_launcher = find_launcher(
            ((cuda_gpu(mem="12G")*2) if gpu else cpu()) & req_duration, tags=tags
        )
    else:
        assert gpu, "Running full scale experiment without GPU is not recommended"
        gpu_launcher = find_launcher((cuda_gpu(mem="48G")) & req_duration, tags=tags)
        gpu_launcher_4 = find_launcher(cuda_gpu(mem="4G") & req_duration, tags=tags)

    # Sets the working directory and the name of the xp
    name = configuration.type

    with experiment(workdir, name, host=host, port=port, launcher=launcher) as xp:
        # Needed by Pyserini
        xp.setenv("JAVA_HOME", find_java_home())

        # Misc
        device = CudaDevice() if gpu else Device()
        random = Random(seed=0)
        basemodel = BM25()
        # create a random scorer as the most naive baseline
        random_scorer = RandomScorer(random=random).tag("model", "random")
        measures = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10]

        # Creates the directory with tensorboard data
        runs_path = xp.resultspath / "runs"
        cleanupdir(runs_path)
        runs_path.mkdir(exist_ok=True, parents=True)
        logging.info("Monitor learning with:")
        logging.info("tensorboard --logdir=%s", runs_path)

        # Datasets: train, validation and test
        documents = prepare_dataset("irds.msmarco-passage.documents") # for indexing 
        cars_documents = prepare_dataset("irds.car.v1.5.documents") # for indexing
        devsmall = prepare_dataset("irds.msmarco-passage.dev.small")

        # We will evaluate on TREC DL 2019 and 2020
        tests: EvaluationsCollection = EvaluationsCollection(
            trec2019=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2019"), measures
            ),
            trec2020=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2020"), measures
            ),
            msmarco_dev=Evaluations(devsmall, measures),
            # trec_car=Evaluations(prepare_dataset("irds.car.v1.5.test200"), measures),
        )

        # Build the MS Marco index and definition of first stage rankers
        index = IndexCollection(documents=documents, storeContents=True).submit()
        base_retriever_ms = AnseriniRetriever(k=topK, index=index, model=basemodel)

        # Build the TREC CARS index
        index_cars = IndexCollection(documents=cars_documents, storeContents=True).submit(launcher = cpu_launcher_4g)
        base_retriever_cars = AnseriniRetriever(k=topK, index=index_cars, model=basemodel)

        base_retrievers = {
            'irds.car.v1.5.documents@irds': base_retriever_cars,
            'irds.msmarco-passage.documents@irds': base_retriever_ms
        }

        # Search and evaluate with BM25
        bm25_retriever_ms = AnseriniRetriever(k=topK, index=index, model=basemodel).tag(
            "model", "bm25"
        )
        bm25_retriever_car = AnseriniRetriever(k=topK, index=index_cars, model=basemodel).tag(
            "model", "bm25"
        )

        bm25_retriever = {
            'irds.car.v1.5.documents@irds': bm25_retriever_car,
            'irds.msmarco-passage.documents@irds': bm25_retriever_ms
        }


        # FIXME: Resolve the OoM by using a GPU. Recommend to rewrite the code in the ir_measures
        # Evaluate BM25 as well as the random scorer (low baseline)
        tests.evaluate_retriever(
            lambda documents: bm25_retriever[documents.id], 
            gpu_launcher_4
        )

        tests.evaluate_retriever(
            lambda documents: random_scorer.getRetriever(
                base_retrievers[documents.id], batch_size, PowerAdaptativeBatcher()
            ),
            gpu_launcher_4
        )

        # define the trainer for monobert

        monobert_scorer = CrossScorerHuggingface(
            # TODO: implement the CrossScorerHuggingface
            ...
        ).tag("model", "monobert-large")

        tests.evaluate_retriever(
            lambda documents: monobert_scorer.getRetriever(
                base_retrievers[documents.id], batch_size, PowerAdaptativeBatcher(), device = device
            ),
            gpu_launcher
        )

        # Waits that experiments complete
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
