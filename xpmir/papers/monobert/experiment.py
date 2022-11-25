# Implementation of the experiments in the paper by training with msmarcos
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
from xpmir.text.huggingface import DualDuoBertTransformerEncoder, DualTransformerEncoder
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

    # Number of topics in the validation set
    VAL_SIZE = configuration.Learner.validation_size

    # Number of batches per epoch (# samples = STEPS_PER_EPOCH * batch_size)
    STEPS_PER_EPOCH = configuration.Learner.steps_per_epoch

    # Validation interval (in epochs)
    validation_interval = configuration.Learner.validation_interval

    # How many document to re-rank for the monobert
    topK = configuration.Retriever_mono.k
    # How many documents to use for cross-validation in monobert
    valtopK = configuration.Retriever_mono.val_k

    # The numbers of the warmup steps during the training
    num_warmup_steps = configuration.Learner.num_warmup_steps

    # Our default launcher for light tasks
    req_duration = duration("6 days")
    launcher = find_launcher(cpu() & req_duration, tags=tags)

    batch_size = batch_size or configuration.Learner.batch_size
    max_epochs = max_epochs or configuration.Learner.max_epoch

    # FIXME: uses CPU constraints rather than GPU
    cpu_launcher_4G = find_launcher(cuda_gpu(mem="4G"))

    if configuration.type == "monobert-small":
        # We request a GPU, and if none, a CPU
        gpu_launcher = find_launcher(
            ((cuda_gpu(mem="12G")*2) if gpu else cpu()) & req_duration, tags=tags
        )
    else:
        assert gpu, "Running full scale experiment without GPU is not recommended"
        gpu_launcher = find_launcher((cuda_gpu(mem="48G")) & req_duration, tags=tags)

    logging.info(
        f"Number of epochs {max_epochs}, validation interval {validation_interval}"
    )

    assert (
        max_epochs % validation_interval == 0
    ), f"Number of epochs ({max_epochs}) is not a multiple of validation interval ({validation_interval})"

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
        dev = prepare_dataset("irds.msmarco-passage.dev")
        ds_val = RandomFold(
            dataset=dev, seed=123, fold=0, sizes=[VAL_SIZE], exclude=devsmall.topics
        ).submit()

        # We will evaluate on TREC DL 2019 and 2020
        tests: EvaluationsCollection = EvaluationsCollection(
            trec2019=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2019"), measures
            ),
            trec2020=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2020"), measures
            ),
            msmarco_dev=Evaluations(devsmall, measures),
            # works not well on it shows the monobert is weak on zero-shot
            # trec_car=Evaluations(prepare_dataset("irds.car.v1.5.test200"), measures),
        )

        # Build the MS Marco index and definition of first stage rankers
        index = IndexCollection(documents=documents, storeContents=True).submit()
        base_retriever_ms = AnseriniRetriever(k=topK, index=index, model=basemodel)
        base_retriever_ms_val = AnseriniRetriever(k=valtopK, index=index, model=basemodel)

        # Build the TREC CARS index
        index_cars = IndexCollection(documents=cars_documents, storeContents=True).submit(launcher = cpu_launcher_4G)
        base_retriever_cars = AnseriniRetriever(k=topK, index=index_cars, model=basemodel)
        base_retriever_cars_val = AnseriniRetriever(k=valtopK, index=index_cars, model=basemodel)

        base_retrievers = {
            'irds.car.v1.5.documents@irds': base_retriever_cars,
            'irds.msmarco-passage.documents@irds': base_retriever_ms
        }

        base_retrievers_val = {
            'irds.car.v1.5.documents@irds': base_retriever_cars_val,
            'irds.msmarco-passage.documents@irds': base_retriever_ms_val
        }

        factory_retriever = lambda scorer, documents: scorer.getRetriever(
            base_retrievers[documents.id], batch_size, PowerAdaptativeBatcher(), device=device
        )

        factory_retriever_val = lambda scorer, documents: scorer.getRetriever(
            base_retrievers_val[documents.id], batch_size, PowerAdaptativeBatcher(), device=device
        )

        # Defines how we sample train examples
        # (using the shuffled pre-computed triplets from MS Marco)
        train_triples = prepare_dataset("irds.msmarco-passage.train.docpairs")
        triplesid = ShuffledTrainingTripletsLines(
            seed=123,
            data=train_triples,
        ).submit()
        train_sampler = TripletBasedSampler(source=triplesid, index=index)

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


        # FIXME: Resolve the OoM by using a GPU for devsmall. Recommend to rewrite the code in the ir_measures
        # Evaluate BM25 as well as the random scorer (low baseline)
        tests.evaluate_retriever(
            lambda documents: bm25_retriever[documents.id], 
            cpu_launcher_4G
        )

        tests.evaluate_retriever(
            lambda documents: random_scorer.getRetriever(
                bm25_retriever[documents.id], batch_size, PowerAdaptativeBatcher()
            ),
            cpu_launcher_4G
        )

        # define the trainer for monobert
        monobert_trainer = pairwise.PairwiseTrainer(
            lossfn=pairwise.PointwiseCrossEntropyLoss().tag("loss", "pce"),
            sampler=train_sampler,
            batcher=PowerAdaptativeBatcher(),
            batch_size=batch_size,
        )

        scheduler = LinearWithWarmup(num_warmup_steps=num_warmup_steps, min_factor=configuration.Learner.warmup_min_factor)

        monobert_scorer = CrossScorer(
            encoder=DualTransformerEncoder(trainable=True, maxlen=512, dropout=0.1)
        ).tag("model", "monobert")

        monobert_reranking = RerankingPipeline(
            monobert_trainer,
            ParameterOptimizer(
                scheduler=scheduler,
                optimizer=Adam(lr=configuration.Learner.lr, weight_decay=1e-2),
            ),
            factory_retriever,
            STEPS_PER_EPOCH,
            max_epochs,
            ds_val,
            {"RR@10": True, "AP": False, "nDCG": False},
            tests,
            device=device,
            validation_retriever_factory=factory_retriever_val,
            validation_interval=validation_interval,
            launcher=gpu_launcher,
            evaluate_launcher=gpu_launcher,
            runs_path=runs_path,

            # FIXME: The write and read of the model in the DataParallel has a .module. to resolve the problem,
            # It should be better to create a adapter for the DistributedHook class for the DataParallel and 
            # rewrite the function read and load
            # Further, it should be better if we can implement the hooks in the form like 
            # (models=[monobert_scorer]) where can paralize more things

            # hooks=[
            #     setmeta(DistributedHook(models=[monobert_scorer.encoder]), True)
            # ]
        )

        # Run the monobert and use the result as the baseline for duobert
        monobert_reranking.run(monobert_scorer)

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