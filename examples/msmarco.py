import logging
from pathlib import Path
import os

from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from experimaestro import experiment, tag, tagspath
from experimaestro.click import click, forwardoption
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions
from experimaestro.utils import cleanupdir
from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluate
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor import Device, Random
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.optim import Adam
from xpmir.letor.samplers import ModelBasedSampler, Sampler, TripletBasedSampler
from xpmir.letor.trainers import Trainer
import xpmir.letor.trainers.pairwise as pairwise
from xpmir.neural.drmm import Drmm
from xpmir.neural.colbert import Colbert
from xpmir.neural.jointclassifier import JointClassifier
from xpmir.rankers import RandomScorer, Scorer, TwoStageRetriever
from xpmir.rankers.standard import BM25
from xpmir.vocab.huggingface import DualTransformerEncoder, TransformerVocab
from xpmir.vocab.wordvec_vocab import WordvecUnkVocab

logging.basicConfig(level=logging.INFO)


def evaluate(token=None, launcher=None, **kwargs):
    v = Evaluate(measures=["AP", "P@20", "NDCG", "NDCG@20", "RR", "RR@10"], **kwargs)
    if token is not None:
        v = token(1, v)
    return v.submit(launcher=launcher)


# --- Experiment
@forwardoption.max_epoch(Learner, default=None)
@click.option(
    "--scheduler", type=click.Choice(["slurm"]), help="Use a scheduler (slurm)"
)
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option(
    "--batch-size", type=int, default=None, help="Batch size (validation and test)"
)
@click.option("--small", is_flag=True, help="Use small datasets")
@click.option(
    "--grad-acc-batch",
    default=0,
    help="Micro-batch size when training BERT-based models",
)
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.command()
def cli(
    debug, small, scheduler, gpu, port, workdir, max_epoch, grad_acc_batch, batch_size
):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    # Number of topics in the validation set
    VAL_SIZE = 500

    # Number of batches per epoch (# samples = BATCHES_PER_EPOCH * batch_size)
    BATCHES_PER_EPOCH = 32

    # Validation interval (in epochs)
    validation_interval = 16

    # How many document to re-rank
    topK = 100
    # How many documents to use for cross-validation
    valtopK = 100

    if small:
        VAL_SIZE = 10
        validation_interval = 1
        topK = 20
        valtopK = 20
        batch_size = batch_size or 16
        max_epoch = max_epoch or 4
    else:
        batch_size = batch_size or 256
        max_epoch = max_epoch or 64

    logging.info(
        f"Number of epochs {max_epoch}, validation interval {validation_interval}"
    )

    assert (
        max_epoch % validation_interval == 0
    ), f"Number of epochs ({max_epoch}) is not a multiple of validation interval ({validation_interval})"

    # Sets the working directory and the name of the xp
    if scheduler == "slurm":
        import socket

        host = socket.getfqdn()
        launcher = SlurmLauncher()
        gpulauncher = launcher.config(gpus=1) if gpu else launcher
    else:
        host = None
        launcher = None
        gpulauncher = None

    name = "msmarco-small" if small else "msmarco"
    with experiment(workdir, name, host=host, port=port, launcher=launcher) as xp:
        if gpulauncher:
            gpulauncher.setNotificationURL(launcher.notificationURL)
        if scheduler is None:
            token = xp.token("main", 1)
        else:

            def token(value, task):
                return task

        xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])

        # Misc
        device = Device(gpu=gpu)
        random = Random(seed=0)
        wordembs = prepare_dataset("edu.stanford.glove.6b.50")
        glove = WordvecUnkVocab(data=wordembs, random=random)

        # Train / validation / test
        documents = prepare_dataset("irds.msmarco-passage.documents")
        train_triples = prepare_dataset("irds.msmarco-passage.train.docpairs")
        devsmall = prepare_dataset("irds.msmarco-passage.dev.small")
        dev = prepare_dataset("irds.msmarco-passage.dev")
        ds_val = RandomFold(
            dataset=dev, seed=123, fold=0, sizes=[VAL_SIZE], exclude=devsmall.topics
        ).submit()
        triplesid = ShuffledTrainingTripletsLines(
            seed=123,
            data=train_triples,
        ).submit()

        tests = {
            "trec2019": prepare_dataset("irds.msmarco-passage.trec-dl-2019"),
            "msmarco-dev": devsmall,
        }

        # MS Marco index
        index = IndexCollection(documents=documents, storeContents=True).submit()
        test_index = index

        # Base models
        basemodel = BM25()
        random_scorer = RandomScorer(random=random).tag("model", "random")

        # Creates the validation dataset

        # This part is used for validation

        train_sampler = TripletBasedSampler(source=triplesid, index=index)

        # Base retrievers
        base_retriever = AnseriniRetriever(k=topK, index=index, model=basemodel)
        base_retriever_val = AnseriniRetriever(k=valtopK, index=index, model=basemodel)

        # Search and evaluate with BM25
        bm25_retriever = AnseriniRetriever(
            k=topK, index=test_index, model=basemodel
        ).tag("model", "bm25")

        evaluations = {}
        for key, test in tests.items():
            evaluations[key] = [
                evaluate(dataset=test, retriever=bm25_retriever),
                evaluate(
                    dataset=test,
                    retriever=base_retriever.getReranker(random_scorer, batch_size),
                ),
            ]

        # @lru_cache
        def trainer(lr=1e-3, grad_acc_batch=0, lossfn=None):
            return pairwise.PairwiseTrainer(
                optimizer=Adam(lr=lr),
                device=device,
                lossfn=lossfn or pairwise.PointwiseCrossEntropyLoss(),
                batches_per_epoch=BATCHES_PER_EPOCH,
                sampler=train_sampler,
                grad_acc_batch=grad_acc_batch,
                batch_size=batch_size,
            )

        # Train and evaluate with each model
        runspath = xp.resultspath / "runs"
        cleanupdir(runspath)
        runspath.mkdir(exist_ok=True, parents=True)

        def run(scorer: Scorer, trainer: Trainer):

            validation = ValidationListener(
                dataset=ds_val,
                retriever=base_retriever_val.getReranker(scorer, valtopK),
                validation_interval=validation_interval,
                metrics={"RR@10": True, "AP": False},
            )

            learner = Learner(
                trainer=trainer,
                random=random,
                scorer=scorer,
                max_epoch=tag(max_epoch),
                listeners={"bestval": validation},
            )
            outputs = token(1, learner).submit(launcher=gpulauncher)
            (runspath / tagspath(learner)).symlink_to(learner.logpath)

            # Evaluate the neural model
            for key, test in tests.items():
                best = outputs["listeners"]["bestval"]["RR@10"]

                evaluations[key].append(
                    evaluate(
                        token=token,
                        dataset=test,
                        retriever=base_retriever.getReranker(best, batch_size),
                        launcher=gpulauncher,
                    )
                )

        # Compares PCE and Softmax
        for lossfn in (
            pairwise.PointwiseCrossEntropyLoss().tag("loss", "pce"),
            pairwise.SoftmaxLoss().tag("loss", "softmax"),
        ):

            # DRMM
            drmm = Drmm(vocab=glove, add_runscore=False, index=index).tag(
                "model", "drmm"
            )
            run(drmm, trainer(lr=tag(1e-2), lossfn=lossfn))

            # We use micro-batches of size 8 for BERT-based models
            # colbert = Colbert(vocab=TransformerVocab(trainable=True), dlen=512).tag(
            #     "model", "colbert"
            # )
            # run(colbert, trainer(lr=tag(1e-3), grad_acc_batch=2))

            colbert = Colbert(
                vocab=TransformerVocab(trainable=True),
                masktoken=False,
                doctoken=False,
                querytoken=False,
                dlen=512,
            ).tag("model", "colbert")
            for lr in [1e-6, 1e-4]:
                run(
                    colbert,
                    trainer(lr=tag(lr), grad_acc_batch=grad_acc_batch, lossfn=lossfn),
                )

            # Vanilla bert
            dual = JointClassifier(encoder=DualTransformerEncoder(trainable=True)).tag(
                "model", "dual"
            )
            run(
                dual,
                trainer(lr=tag(1e-4), grad_acc_batch=grad_acc_batch, lossfn=lossfn),
            )

        # Wait that experiments complete
        xp.wait()

        for key, dsevaluations in evaluations.items():
            print(f"=== {key}")
            for evaluation in dsevaluations:
                print(
                    f"Results for {evaluation.__xpm__.tags()}\n{evaluation.results.read_text()}\n"
                )


if __name__ == "__main__":
    cli()
