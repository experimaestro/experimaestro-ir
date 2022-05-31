import logging
from pathlib import Path
import os
from typing import List, Optional

from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from experimaestro import experiment, tag, tagspath, copyconfig, setmeta
from experimaestro.click import click, forwardoption
from experimaestro.launchers.slurm import SlurmLauncher
from experimaestro.utils import cleanupdir
from xpmir.datasets.adapters import RandomFold, RetrieverBasedCollection
from xpmir.evaluation import Evaluate
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor.devices import CudaDevice, Device
from xpmir.letor import Random
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.optim import Adamp
from xpmir.letor.samplers import PairwiseInBatchNegativesSampler, TripletBasedSampler
from xpmir.letor.schedulers import CosineWithWarmup, LinearWithWarmup
from xpmir.index.faiss import IndexBackedFaiss, FaissRetriever, FlatIPIndexer
from xpmir.index.sparse import (
    SparseRetriever,
    SparseRetrieverIndex,
    SparseRetrieverIndexBuilder,
)
from xpmir.letor.trainers import Trainer, pointwise
from xpmir.letor.trainers.batchwise import BatchwiseTrainer, SoftmaxCrossEntropy
from xpmir.letor.batchers import PowerAdaptativeBatcher
import xpmir.letor.trainers.pairwise as pairwise
from xpmir.neural.dual import DenseDocumentEncoder, DenseQueryEncoder, DotDense
from xpmir.rankers import FullRetriever, RandomScorer, Scorer
from xpmir.letor.optim import ParameterOptimizer
from xpmir.rankers.standard import BM25
from xpmir.neural.splade import DistributedSpladeTextEncoderHook, spladeV2
from xpmir.measures import AP, P, nDCG, RR
from xpmir.neural.pretrained import tas_balanced
from xpmir.text.huggingface import DistributedModelHook, TransformerEncoder

logging.basicConfig(level=logging.INFO)


def evaluate(token=None, launcher=None, **kwargs):
    v = Evaluate(
        measures=[AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10], **kwargs
    )
    if token is not None:
        v = token(1, v)
    return v.submit(launcher=launcher)


# --- Experiment
@forwardoption.max_epochs(Learner, default=None)
@click.option(
    "--scheduler", type=click.Choice(["slurm"]), help="Use a scheduler (slurm)"
)
@click.option("--slurm-gpu-account", help="Slurm account for GPU", type=str)
@click.option(
    "--slurm-gpu-time", help="Slurm time for GPU", type=str, default="5-00:00:00"
)
@click.option(
    "--slurm-gpu-qos", help="Slurm QOS for GPU", type=str, default="5-00:00:00"
)
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option(
    "--batch-size", type=int, default=None, help="Batch size (validation and test)"
)
@click.option("--small", is_flag=True, help="Use small datasets")
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.option("--fqdn", is_flag=True, help="Use qualified host name")
@click.argument("workdir", type=Path)
@click.command()
def cli(
    debug: bool,
    small: bool,
    scheduler: Optional[str],
    gpu: bool,
    port: int,
    workdir: str,
    max_epochs: int,
    batch_size: Optional[int],
    fqdn: bool,
    slurm_gpu_account: Optional[str],
    slurm_gpu_time: str,
    slurm_gpu_qos: Optional[str],
):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    if max_epochs is not None:
        max_epochs = int(max_epochs)

    # Number of topics in the validation set
    VAL_SIZE = 500

    # Number of batches per epoch (# samples = BATCHES_PER_EPOCH * batch_size)
    BATCHES_PER_EPOCH = 32

    if small:
        VAL_SIZE = 10
        validation_interval = 1
        topK = 20
        valtopK = 20
        batch_size = batch_size or 16
        max_epochs = max_epochs or 4

        batch_size = 16
        splade_batch_size = 16

        retTopK = 10

        # Validation interval (in epochs)
        validation_interval = max_epochs // 4

        early_stop = 0

    else:
        # How many document to re-rank
        topK = 1000

        # How many documents to use for cross-validation
        valtopK = 200

        # 2^14 epochs x 32 batch/epoch = 500K iterations
        max_epochs = max_epochs or 2**14

        # Batch size
        splade_batch_size = batch_size or 48
        batch_size = batch_size or 124

        # Validation interval (every 256 epochs = 256 epoch * 128 steps/epoch = every 32768 steps)
        validation_interval = 2**8

        # Stop training if no improvement after 16 evaluations
        early_stop = validation_interval * 16

        # Top-K when building
        retTopK = 50

    # Try to have just one batch at inference
    test_batch_size = topK
    max_epochs = int(max_epochs)

    logging.info(
        f"Number of epochs {max_epochs}, validation interval {validation_interval}"
    )

    if (max_epochs % validation_interval) != 0:
        raise AssertionError(
            f"Number of epochs ({max_epochs}) is not a multiple of validation interval ({validation_interval})"
        )

    # Sets the working directory and the name of the xp
    if scheduler == "slurm":
        import socket

        host = socket.getfqdn() if fqdn else socket.gethostname()
        launcher = SlurmLauncher()
        # slurm: 1 GPU, 5 days time limit
        gpulauncher = (
            launcher.config(
                gpus=1,
                qos=slurm_gpu_qos,
                account=slurm_gpu_account,
                time=slurm_gpu_time,
            )
            if gpu
            else launcher
        )
        gpulauncher2x = (
            launcher.config(
                gpus=2,
                qos=slurm_gpu_qos,
                account=slurm_gpu_account,
                time=slurm_gpu_time,
            )
            if gpu
            else launcher
        )
        gpulauncher4x = (
            launcher.config(
                gpus=4,
                qos=slurm_gpu_qos,
                account=slurm_gpu_account,
                time=slurm_gpu_time,
            )
            if gpu
            else launcher
        )
        # GPU launcher with a lot of memory
        gpulauncher_mem64 = gpulauncher.config(gpus=2, mem="64G") if gpu else launcher
    else:
        host = None
        launcher = None
        gpulauncher = None
        gpulauncher_mem64 = None
        gpulauncher2x = None

    name = "splade-small" if small else "splade"
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
        device = CudaDevice() if gpu else Device()
        random = Random(seed=0)

        # Get a sparse retriever from a dual scorer

        # Train / validation / test
        documents = prepare_dataset("irds.msmarco-passage.documents")

        def sparse_retriever(scorer, documents):
            index = SparseRetrieverIndexBuilder(
                batch_size=512,
                batcher=PowerAdaptativeBatcher(),
                encoder=DenseDocumentEncoder(scorer=scorer),
                device=device,
                documents=documents,
            ).submit(launcher=gpulauncher)
            return SparseRetriever(
                index=index,
                topk=test_batch_size,
                encoder=DenseQueryEncoder(scorer=scorer),
            )

        index = IndexCollection(documents=documents, storeContents=True).submit()

        train_triples = prepare_dataset("irds.msmarco-passage.train.docpairs")
        dev = prepare_dataset("irds.msmarco-passage.dev")
        triplesid = ShuffledTrainingTripletsLines(
            seed=123,
            data=train_triples,
        ).submit()

        # Development
        devsmall = prepare_dataset("irds.msmarco-passage.dev.small")

        # Build a dev. collection for full-ranking
        # "Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling"

        tasb = tas_balanced()
        tasb_index = IndexBackedFaiss(
            indexer=FlatIPIndexer(),
            device=device,
            normalize=False,
            documents=documents,
            encoder=DenseDocumentEncoder(scorer=tasb),
            batchsize=32768,
            batcher=PowerAdaptativeBatcher(),
            hooks=[
                setmeta(
                    DistributedModelHook(transformer=tasb.query_encoder.encoder), True
                )
            ],
        ).submit(launcher=gpulauncher_mem64)
        tasb_retriever = FaissRetriever(
            index=tasb_index, topk=retTopK, encoder=DenseQueryEncoder(scorer=tasb)
        )

        tests = {
            "trec2019": prepare_dataset("irds.msmarco-passage.trec-dl-2019"),
            "trec2020": prepare_dataset("irds.msmarco-passage.trec-dl-2020"),
            "msmarco-dev": devsmall,
        }

        # MS Marco index
        test_index = index

        # Base models
        basemodel = BM25()

        # Creates the validation dataset
        train_sampler = TripletBasedSampler(source=triplesid, index=index)

        # Search and evaluate with BM25
        bm25_retriever = AnseriniRetriever(k=topK, index=test_index, model=basemodel)

        # Build collection from TAS-Balanced
        ds_val = RetrieverBasedCollection(
            dataset=RandomFold(
                dataset=dev, seed=123, fold=0, sizes=[VAL_SIZE], exclude=devsmall.topics
            ).submit(),
            retrievers=[
                tasb_retriever,
                AnseriniRetriever(k=retTopK, index=test_index, model=basemodel),
            ],
        ).submit(launcher=gpulauncher_mem64)

        # Base retrievers for validation
        base_retriever_val = FullRetriever(documents=ds_val.documents)

        evaluations = {}
        for key, test in tests.items():
            evaluations[key] = [
                evaluate(
                    dataset=test,
                    retriever=copyconfig(bm25_retriever).tag("model", "bm25"),
                ),
            ]

        # Train and evaluate with each model
        runspath = xp.resultspath / "runs"
        cleanupdir(runspath)
        runspath.mkdir(exist_ok=True, parents=True)

        def run(
            scorer: Scorer,
            trainer: Trainer,
            optimizers: List,
            create_retriever,
            hooks=[],
            launcher=gpulauncher,
        ):
            # Learn the model
            validation = ValidationListener(
                dataset=ds_val,
                retriever=base_retriever_val.getReranker(
                    scorer, valtopK, PowerAdaptativeBatcher()
                ),
                early_stop=early_stop,
                validation_interval=validation_interval,
                metrics={"RR@10": True, "AP": False, "nDCG@10": False},
            )

            learner = Learner(
                trainer=trainer,
                use_fp16=True,
                random=random,
                scorer=scorer,
                device=device,
                optimizers=optimizers,
                max_epochs=tag(max_epochs),
                listeners={"bestval": validation},
                hooks=hooks,
            )
            outputs = token(1, learner).submit(launcher=launcher)
            (runspath / tagspath(learner)).symlink_to(learner.logpath)

            # Evaluate the neural model
            for key, test in tests.items():
                # Build the retrieval
                best = outputs["listeners"]["bestval"]["RR@10"]
                retriever = create_retriever(best)

                evaluations[key].append(
                    evaluate(
                        token=token,
                        dataset=test,
                        retriever=retriever,
                        launcher=gpulauncher,
                    )
                )

        # Launch Splade
        def reranker_retriever(best):
            return base_retriever.getReranker(
                best, test_batch_size, device=device, batcher=PowerAdaptativeBatcher()
            )

        def faiss_retriever(best):
            tasb_index = IndexBackedFaiss(
                indexer=FlatIPIndexer(),
                device=device,
                normalize=False,
                documents=documents,
                encoder=DenseDocumentEncoder(scorer=best),
                batchsize=1024,
                batcher=PowerAdaptativeBatcher(),
            ).submit()
            return FaissRetriever(
                index=tasb_index, topk=10, encoder=DenseQueryEncoder(scorer=tasb)
            )

        distilbert = tag("distilbert-base-uncased")

        ibn_sampler = PairwiseInBatchNegativesSampler(sampler=train_sampler)
        scheduler = CosineWithWarmup(num_warmup_steps=1000)

        # Train splade
        spladev2, flops = spladeV2(1e-1, 1e-1)

        batchwise_trainer_flops = BatchwiseTrainer(
            batch_size=splade_batch_size,
            sampler=ibn_sampler,
            lossfn=SoftmaxCrossEntropy(),
            hooks=[flops],
        )
        run(
            spladev2.tag("model", "splade-v2"),
            batchwise_trainer_flops,
            [ParameterOptimizer(scheduler=scheduler, optimizer=Adam(lr=2e-5))],
            lambda scorer: sparse_retriever(scorer, documents),
            hooks=[
                setmeta(DistributedSpladeTextEncoderHook(splade=spladev2.encoder), True)
            ],
            launcher=gpulauncher4x,
        )

        # Train Dense

        distilbert_encoder = TransformerEncoder(model_id=distilbert, trainable=True)
        siamese = DotDense(
            query_encoder=distilbert_encoder.with_maxlength(30),
            encoder=distilbert_encoder.with_maxlength(200),
        ).tag("model", "siamese")
        adam_7 = [
            ParameterOptimizer(
                optimizer=Adam(lr=tag(7e-6)).tag("optim", "Adam"),
                scheduler=CosineWithWarmup(num_warmup_steps=1000),
            )
        ]
        batchwise_trainer = BatchwiseTrainer(
            batch_size=batch_size, sampler=ibn_sampler, lossfn=SoftmaxCrossEntropy()
        )
        run(siamese, batchwise_trainer, adam_7, faiss_retriever, launcher=gpulauncher2x)

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
