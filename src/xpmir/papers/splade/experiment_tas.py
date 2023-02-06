import logging
from pathlib import Path
import os
from typing import Dict

from omegaconf import OmegaConf
from datamaestro import prepare_dataset
from experimaestro.launcherfinder import find_launcher

from experimaestro import experiment, copyconfig, setmeta
from experimaestro.click import click
from xpmir.configuration import omegaconf_argument
from xpmir.distributed import DistributedHook
from xpmir.documents.samplers import RandomDocumentSampler
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor.devices import CudaDevice, Device
from xpmir.letor import Random
from xpmir.index.faiss import IndexBackedFaiss, FaissRetriever
from xpmir.rankers import RandomScorer
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.neural.dual import DenseDocumentEncoder, DenseQueryEncoder
from xpmir.rankers.standard import BM25
from xpmir.measures import AP, P, nDCG, RR
from xpmir.models import AutoModel

logging.basicConfig(level=logging.INFO)

# @forwardoption.max_epochs(Learner, default=None)
# @click.option("--tags", type=str, default="", help="Tags for selecting the launcher")
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option(
    "--env", help="Define one environment variable", type=(str, str), multiple=True
)
# @click.option("--gpu", is_flag=True, help="Use GPU")
# @click.option(
#     "--batch-size", type=int, default=None, help="Batch size (validation and test)"
# )
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
    assert configuration.Launcher.gpu, "It is recommend to do this on GPU"
    cpu_launcher_index = find_launcher(configuration.Indexation.requirements)
    gpu_launcher_index = find_launcher(configuration.Indexation.requirements, tags=tags)
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
        documents_trec_covid = prepare_dataset(
            "irds.beir.trec-covid.documents"
        )  # the dataset for trec_covid

        # Index for msmarcos
        index_trec_covid = IndexCollection(
            documents=documents_trec_covid, storeContents=True
        ).submit()

        # Build a dev. collection for full-ranking (validation)
        # "Efficiently Teaching an Effective Dense Retriever with Balanced Topic Aware Sampling"
        tasb = AutoModel.load_from_hf_hub(
            "xpmir/tas-balanced"
        )  # create a scorer from huggingface

        random_scorer = RandomScorer(random=random).tag("model", "random")

        # task to train the tas_balanced encoder for the document list and generate an index for retrieval
        tasb_index = IndexBackedFaiss(
            indexspec=configuration.tas_balance_retriever.indexspec,
            device=device,
            normalize=False,
            documents=documents_trec_covid,
            sampler=RandomDocumentSampler(
                documents=documents_trec_covid,
                max_count=configuration.tas_balance_retriever.faiss_max_traindocs,
                random=random,
            ),  # Just use a fraction of the dataset for training
            encoder=DenseDocumentEncoder(scorer=tasb),
            batchsize=2048,
            batcher=PowerAdaptativeBatcher(),
            # hooks=[setmeta(DistributedHook(models=[tasb]), True)],
        ).submit(launcher=gpu_launcher_index)

        tasb_index_hnsw = IndexBackedFaiss(
            indexspec="HNSW",
            device=device,
            normalize=False,
            documents=documents_trec_covid,
            sampler=None,
            encoder=DenseDocumentEncoder(scorer=tasb),
            batchsize=2048,
            batcher=PowerAdaptativeBatcher(),
            # hooks=[setmeta(DistributedHook(models=[tasb]), True)],
        ).submit(launcher=gpu_launcher_index)

        # A retriever if tas-balanced. We use the index of the faiss.
        # Used it to create the validation dataset.
        tasb_retriever = FaissRetriever(
            index=tasb_index, topk=20, encoder=DenseQueryEncoder(scorer=tasb)
        )
        tasb_retriever_2 = FaissRetriever(
            index=tasb_index_hnsw, topk=20, encoder=DenseQueryEncoder(scorer=tasb)
        )

        # also the bm25 for creating the validation set.
        basemodel = BM25()

        # define the evaluation measures and dataset.
        measures = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10]
        tests: EvaluationsCollection = EvaluationsCollection(
            trec_covid=Evaluations(prepare_dataset("irds.beir.trec-covid"), measures)
        )

        # compute the baseline performance on the test dataset.
        # Bm25
        bm25_retriever_trec = AnseriniRetriever(
            k=20, index=index_trec_covid, model=basemodel
        )

        tests.evaluate_retriever(
            random_scorer.getRetriever(
                bm25_retriever_trec, 20, PowerAdaptativeBatcher()
            ),
            gpu_launcher_index,
        )

        tests.evaluate_retriever(copyconfig(bm25_retriever_trec).tag("model", "bm25"))

        # tas-balance
        tests.evaluate_retriever(
            copyconfig(tasb_retriever).tag("model", "tasb"), gpu_launcher_index
        )
        tests.evaluate_retriever(
            copyconfig(tasb_retriever_2).tag("model", "tasb-hnsw"), gpu_launcher_index
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
