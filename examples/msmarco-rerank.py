#
# This files contains an example of an experiment that
# trains second stage rankers (based on BM25) on MS Marco
#
# Trained and evaluated models:
# - ColBERT
# - BERT
# - DRMM (with Glove)
#
# Compares PCE and Softmax

import logging
from pathlib import Path
from typing import List
from datamaestro import prepare_dataset

from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.letor.devices import CudaDevice

from experimaestro.launcherfinder import cpu, cuda_gpu, find_launcher
from experimaestro import experiment, tag, tagspath
from experimaestro.click import click, forwardoption
from experimaestro.launcherfinder.specs import duration
from experimaestro.utils import cleanupdir

from xpmir.utils import find_java_home
from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluate
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor import Device, Random
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.optim import Adam, AdamW, Optimizer, ParameterOptimizer
from xpmir.letor.samplers import TripletBasedSampler
from xpmir.letor.trainers import Trainer
import xpmir.letor.trainers.pairwise as pairwise
from xpmir.neural.interaction.drmm import Drmm
from xpmir.neural.colbert import Colbert
from xpmir.neural.jointclassifier import JointClassifier
from xpmir.rankers import RandomScorer, Scorer
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import DualTransformerEncoder, TransformerVocab
from xpmir.text.wordvec_vocab import WordvecUnkVocab
from xpmir.measures import AP, P, nDCG, RR

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
    max_epochs = int(max_epochs)

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
        wordembs = prepare_dataset("edu.stanford.glove.6b.50")
        glove = WordvecUnkVocab(data=wordembs, random=random)
        basemodel = BM25()
        random_scorer = RandomScorer(random=random).tag("model", "random")
        measures = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10]

        # Creates the directory with tensorboard data
        runspath = xp.resultspath / "runs"
        cleanupdir(runspath)
        runspath.mkdir(exist_ok=True, parents=True)
        logging.info("Monitor learning with:")
        logging.info("tensorboard --logdir=%s", runspath)

        # Datasets: train, validation and test
        documents = prepare_dataset("irds.msmarco-passage.documents")
        devsmall = prepare_dataset("irds.msmarco-passage.dev.small")
        dev = prepare_dataset("irds.msmarco-passage.dev")
        ds_val = RandomFold(
            dataset=dev, seed=123, fold=0, sizes=[VAL_SIZE], exclude=devsmall.topics
        ).submit()
        # We will evaluate on TREC DL 2019 and 2020, as well as on the msmarco-dev dataset
        tests = {
            "trec2019": prepare_dataset("irds.msmarco-passage.trec-dl-2019"),
            "trec2020": prepare_dataset("irds.msmarco-passage.trec-dl-2020"),
            "msmarco-dev": devsmall,
        }

        # Build the MS Marco index and definition of first stage rankers
        index = IndexCollection(documents=documents, storeContents=True).submit()
        base_retriever = AnseriniRetriever(k=topK, index=index, model=basemodel)
        base_retriever_val = AnseriniRetriever(k=valtopK, index=index, model=basemodel)

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
        evaluations = {}
        for key, test in tests.items():
            evaluations[key] = [
                Evaluate(
                    measures=measures, dataset=test, retriever=bm25_retriever
                ).submit(),
                Evaluate(
                    measures=measures,
                    dataset=test,
                    retriever=base_retriever.getReranker(
                        random_scorer, batch_size, PowerAdaptativeBatcher()
                    ),
                ).submit(),
            ]

        def trainer(lossfn=None):
            return pairwise.PairwiseTrainer(
                lossfn=lossfn or pairwise.PointwiseCrossEntropyLoss(),
                sampler=train_sampler,
                batcher=PowerAdaptativeBatcher(),
                batch_size=batch_size,
            )

        def run(scorer: Scorer, trainer: Trainer, optimizers: List[Optimizer]):
            """Train a scorer with a given trainer, before evaluating"""

            # The validation listener will evaluate the full retriever
            # (1st stage + reranker) and keep the best performing model
            # on the validation set
            validation = ValidationListener(
                dataset=ds_val,
                retriever=base_retriever_val.getReranker(scorer, valtopK),
                validation_interval=validation_interval,
                metrics={"RR@10": True, "AP": False},
            )

            # The learner defines all what is needed
            # to perform several gradient steps
            learner = Learner(
                # Misc settings
                device=device,
                random=random,
                # How to train the model
                trainer=trainer,
                # The model to train
                scorer=scorer,
                # Optimization settings
                steps_per_epoch=STEPS_PER_EPOCH,
                optimizers=optimizers,
                max_epochs=tag(max_epochs),
                # The listeners (here, for validation)
                listeners={"bestval": validation},
            )
            outputs = learner.submit(launcher=gpu_launcher)
            (runspath / tagspath(learner)).symlink_to(learner.logpath)

            # Evaluate the neural model
            for key, test in tests.items():
                best = outputs.listeners["bestval"]["RR@10"]

                evaluations[key].append(
                    Evaluate(
                        measures=measures,
                        dataset=test,
                        retriever=base_retriever.getReranker(
                            best, batch_size, device=device
                        ),
                    ).submit(launcher=gpu_launcher)
                )

        # Compares PCE and Softmax

        optimizers = [ParameterOptimizer(optimizer=AdamW(lr=1e-5, weight_decay=1e-2))]

        for lossfn in (
            pairwise.PointwiseCrossEntropyLoss().tag("loss", "pce"),
            pairwise.SoftmaxLoss().tag("loss", "softmax"),
        ):

            # DRMM
            drmm = Drmm(vocab=glove, index=index).tag("model", "drmm")
            run(drmm, trainer(lossfn=lossfn), optimizers)

            # Train and evaluate Colbert
            colbert = Colbert(
                vocab=TransformerVocab(trainable=True),
                masktoken=False,
                doctoken=False,
                querytoken=False,
                dlen=512,
            ).tag("model", "colbert")
            run(colbert, trainer(lossfn=lossfn), optimizers)

            # Train and evaluate Vanilla BERT
            dual = JointClassifier(encoder=DualTransformerEncoder(trainable=True)).tag(
                "model", "dual"
            )
            run(dual, trainer(lossfn=lossfn), optimizers)

        # ---  End of the experiment

        # Waits that experiments complete
        xp.wait()

        # Display metrics for each trained model
        for key, dsevaluations in evaluations.items():
            print(f"=== {key}")
            for evaluation in dsevaluations:
                print(
                    f"Results for {evaluation.__xpm__.tags()}\n{evaluation.results.read_text()}\n"
                )


if __name__ == "__main__":
    cli()
