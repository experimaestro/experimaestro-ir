#
# This files contains an example of an experiment that
# trains second stage rankers (based on BM25) on MS Marco
#
# Trains and evaluates monoBERT
#
# Compares PCE and Softmax

import dataclasses
import logging
from pathlib import Path

import xpmir.letor.trainers.pairwise as pairwise
from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from xpmir.neural.cross import CrossScorer
from experimaestro import experiment
from experimaestro.click import click, forwardoption
from experimaestro.launcherfinder import cpu, cuda_gpu, find_launcher
from experimaestro.launcherfinder.specs import duration
from experimaestro.utils import cleanupdir
from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor import Device, Random
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.letor.devices import CudaDevice
from xpmir.letor.learner import Learner
from xpmir.letor.optim import AdamW
from xpmir.letor.samplers import TripletBasedSampler
from xpmir.measures import AP, RR, P, nDCG
from xpmir.neural.jointclassifier import JointClassifier
from xpmir.pipelines.reranking import RerankingPipeline
from xpmir.rankers import RandomScorer
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import DualTransformerEncoder
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
@click.option("--small", is_flag=True, help="Use small datasets")
@click.option(
    "--host",
    type=str,
    default=None,
    help="Server hostname (default to localhost, not suitable if your jobs are remote)",
)
@click.option(
    "--port", type=int, default=12345, help="Port for monitoring (default 12345)"
)
@click.argument("workdir", type=Path)
@click.command()
def cli(debug, small, gpu, tags, host, port, workdir, max_epochs, batch_size):
    """Runs an experiment"""
    tags = tags.split(",") if tags else []

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    # Number of topics in the validation set
    VAL_SIZE = 500

    # Number of batches per epoch (# samples = STEPS_PER_EPOCH * batch_size)
    STEPS_PER_EPOCH = 32

    # Validation interval (in epochs)
    validation_interval = 16

    # How many document to re-rank
    topK = 100
    # How many documents to use for cross-validation
    valtopK = 100

    # Our default launcher for light tasks
    req_duration = duration("2 days")
    launcher = find_launcher(cpu() & req_duration, tags=tags)

    if small:
        VAL_SIZE = 10
        validation_interval = 1
        topK = 20
        valtopK = 20
        batch_size = batch_size or 16
        max_epochs = max_epochs or 4

        # We request a GPU, and if none, a CPU
        gpu_launcher = find_launcher(
            (cuda_gpu(mem="4G") if gpu else cpu()) & req_duration, tags=tags
        )
    else:
        assert gpu, "Running full scale experiment without GPU is not recommended"
        batch_size = batch_size or 256
        max_epochs = max_epochs or 64
        gpu_launcher = find_launcher(cuda_gpu(mem="14G") & req_duration, tags=tags)

    logging.info(
        f"Number of epochs {max_epochs}, validation interval {validation_interval}"
    )

    assert (
        max_epochs % validation_interval == 0
    ), f"Number of epochs ({max_epochs}) is not a multiple of validation interval ({validation_interval})"

    # Sets the working directory and the name of the xp
    name = "msmarco-small" if small else "msmarco"
    with experiment(workdir, name, host=host, port=port, launcher=launcher) as xp:
        # Needed by Pyserini
        xp.setenv("JAVA_HOME", find_java_home())

        # Misc
        device = CudaDevice() if gpu else Device()
        random = Random(seed=0)
        basemodel = BM25()
        random_scorer = RandomScorer(random=random).tag("model", "random")
        measures = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10]

        # Creates the directory with tensorboard data
        runs_path = xp.resultspath / "runs"
        cleanupdir(runs_path)
        runs_path.mkdir(exist_ok=True, parents=True)
        logging.info("Monitor learning with:")
        logging.info("tensorboard --logdir=%s", runs_path)

        # Datasets: train, validation and test
        documents = prepare_dataset("irds.msmarco-passage.documents")
        devsmall = prepare_dataset("irds.msmarco-passage.dev.small")
        dev = prepare_dataset("irds.msmarco-passage.dev")
        ds_val = RandomFold(
            dataset=dev, seed=123, fold=0, sizes=[VAL_SIZE], exclude=devsmall.topics
        ).submit()

        # We will evaluate on TREC DL 2019 and 2020, as well as on the msmarco-dev dataset
        tests: EvaluationsCollection = EvaluationsCollection(
            trec2019=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2019"), measures
            ),
            trec2020=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2020"), measures
            ),
            msmarco_dev=Evaluations(devsmall, measures),
        )

        # Build the MS Marco index and definition of first stage rankers
        index = IndexCollection(documents=documents, storeContents=True).submit()
        base_retriever = AnseriniRetriever(k=topK, index=index, model=basemodel)
        base_retriever_val = AnseriniRetriever(k=valtopK, index=index, model=basemodel)

        index_cars = IndexCollection(
            documents=cars_documents, storeContents=True
        ).submit()
        base_retriever_cars = AnseriniRetriever(
            k=topK, index=index_cars, model=basemodel
        )
        base_retrievers = {
            "irds.car.v1.5.documents@irds": base_retriever_cars,
            "irds.msmarco-passage.documents@irds": base_retriever,
        }
        factory = (
            lambda scorer, documents: scorer.getRetriever(
                base_retrievers[documents.id],
                batch_size,
                PowerAdaptativeBatcher(),
                device=device,
            ),
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
        bm25_retriever = AnseriniRetriever(k=topK, index=index, model=basemodel).tag(
            "model", "bm25"
        )

        # Evaluate BM25 as well as the random scorer (low baseline)
        tests.evaluate_retriever(bm25_retriever)
        tests.evaluate_retriever(
            random_scorer.getRetriever(
                bm25_retriever, batch_size, PowerAdaptativeBatcher()
            )
        )

        def trainer(lossfn):
            return pairwise.PairwiseTrainer(
                lossfn=lossfn,
                sampler=train_sampler,
                batcher=PowerAdaptativeBatcher(),
                batch_size=batch_size,
            )

        # Compares PCE and Softmax
        reranker_pce = RerankingPipeline(
            trainer(pairwise.PointwiseCrossEntropyLoss().tag("loss", "pce")),
            AdamW(lr=1e-5, weight_decay=1e-2),
            lambda scorer, documents: scorer.getRetriever(
                base_retriever, batch_size, PowerAdaptativeBatcher(), device=device
            ),
            STEPS_PER_EPOCH,
            max_epochs,
            ds_val,
            {"RR@10": True, "AP": False},
            tests,
            validation_retriever_factory=lambda scorer, documents: scorer.getRetriever(
                base_retriever_val, batch_size, PowerAdaptativeBatcher(), device=device
            ),
            device=device,
            launcher=gpu_launcher,
            evaluate_launcher=gpu_launcher,
            runs_path=runs_path,
        )
        reranker_softmax = dataclasses.replace(
            reranker_pce, trainer=trainer(pairwise.SoftmaxLoss().tag("loss", "softmax"))
        )

        for reranker in [reranker_pce, reranker_softmax]:
            # Train and evaluate Vanilla BERT
            dual = CrossScorer(encoder=DualTransformerEncoder(trainable=True)).tag(
                "model", "dual"
            )
            reranker.run(dual)

        # ---  End of the experiment

        # Waits that experiments complete
        xp.wait()

        # Display metrics for each trained model
        for key, dsevaluations in tests.collection.items():
            print(f"=== {key}")
            for evaluation in dsevaluations.results:
                print(
                    f"Results for {evaluation.__xpm__.tags()}\n{evaluation.results.read_text()}\n"
                )


if __name__ == "__main__":
    cli()
