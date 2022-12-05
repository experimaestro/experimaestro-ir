import logging
from pathlib import Path
import os
from typing import Dict, List, Optional


from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from experimaestro.experimaestro.nodefinder import cpu, cuda_gpu, find_launcher

from experimaestro import experiment, tag, tagspath, copyconfig, setmeta
from experimaestro.click import click, forwardoption
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions
from experimaestro.utils import cleanupdir
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
from xpmir.neural.splade import spladeV2
from xpmir.distributed import DistributedHook
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
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option(
    "--env", help="Define one environment variable", type=(str, str), multiple=True
)
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.option("--fqdn", is_flag=True, help="Use qualified host name")
@click.option(
    "--batch-size", type=int, default=None, help="Batch size (validation and test)"
)
@click.option("--small", is_flag=True, help="Use small datasets")
@click.argument("workdir", type=Path)
@click.command()
def cli(
    env: Dict[str, str],
    debug: bool,
    small: bool,
    port: int,
    workdir: str,
    max_epochs: int,
    batch_size: Optional[int],
    fqdn: bool,
):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    if max_epochs is not None:
        max_epochs = int(max_epochs)

    # Number of topics in the validation set
    VAL_SIZE = 500

    # Number of steps per epoch
    steps_per_epoch = 128

    if small:
        VAL_SIZE = 10
        validation_interval = 1
        topK = 20
        valtopK = 20
        batch_size = batch_size or 16
        max_epochs = max_epochs or 4

        batch_size = 16
        splade_batch_size = 16
        num_warmup_steps = 1000

        # Top-K when building the validation set(tas-balanced)
        retTopK = 10

        # Validation interval (in epochs)
        validation_interval = max_epochs // 4

        early_stop = 0

        # FAISS index building
        indexspec = "OPQ4_16,IVF256_HNSW32,PQ4"
        faiss_max_traindocs = 15_000
    else:
        # How many document to re-rank(use for BM25)
        topK = 1000

        # How many documents to use for cross-validation(for the two-stage reranker)
        valtopK = 200

        # 2^10 epochs x 128 batch/epoch = 130K iterations
        max_epochs = max_epochs or 2**10

        # Validation interval (every 64 epochs = 64 * 128 steps = 8192 steps)
        validation_interval = 2**6

        # Batch size
        splade_batch_size = batch_size or 96
        batch_size = batch_size or 124
        num_warmup_steps = 1000

        # Faiss index spec
        indexspec = "OPQ4_16,IVF65536_HNSW32,PQ4"
        faiss_max_traindocs = 800_000  # around 1/10th of the full dataset

        # Stop training if no improvement after 16 evaluations
        early_stop = validation_interval * 16

        # Top-K when building the validation set(tas-balance)
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

    name = "splade-small" if small else "splade"

    # Launchers
    launcher = find_launcher(cuda_gpu(cpu_memory="2G"))
    gpulauncher_mem48 = find_launcher(
        cuda_gpu(cuda_memory=["48G"]), cuda_gpu(cuda_memory=["24G", "24G"])
    )

    gpu_launcher_mem24 = find_launcher(cuda_gpu("24G"))
    gpulauncher_mem48 = find_launcher((cuda_gpu(mem="24G") * 2 | cuda_gpu(mem="48G")))
    launcher_mem64 = find_launcher(cpu(mem="64G"))

    # Starts the experiment
    with experiment(workdir, name, host=host, port=port, launcher=launcher) as xp:
        # Set environment variables
        xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])
        for key, value in env:
            xp.setenv(key, value)

        # Misc
        device = CudaDevice()
        random = Random(seed=0)

        # Train / validation / test
        documents = prepare_dataset("irds.msmarco-passage.documents")

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
            ).submit(launcher=gpu_launcher_mem24)

            return SparseRetriever(
                index=index,
                topk=test_batch_size,
                batchsize=256,
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
        # TODO: Also make it as a baseline during the evaluation
        tasb = tas_balanced()

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
            hooks=[
                setmeta(
                    DistributedModelHook(transformer=tasb.query_encoder.encoder), True
                )
            ],
        ).submit(launcher=gpu_launcher)

        # A retriever if tas-balanced. We use the index of the faiss.
        # Used it to create the validation dataset.
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

        # Base models --> another baseline
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
        ).submit(launcher=launcher_mem64)

        # Computes the BM25(and random scorer) performance on the validation dataset
        # baseline
        val_index = IndexCollection(
            documents=ds_val.documents, storeContents=True
        ).submit()
        val_bm25_retriever = AnseriniRetriever(k=topK, index=val_index, model=basemodel)
        val_evaluation_bm25 = evaluate(dataset=ds_val, retriever=val_bm25_retriever)
        random_scorer = RandomScorer(random=random).tag("model", "random")
        val_evaluation_random = evaluate(
            dataset=ds_val,
            retriever=random_scorer.getRetriever(val_evaluation_bm25, batch_size=500),
        )

        print(
            f"BM25 on validation: results are stored in {val_evaluation_bm25.results}"
        )
        print(
            f"Random on validation: results are stored in {val_evaluation_random.results}"
        )

        # Base retrievers for validation
        # It retrieve all the document of the collection with score 0
        base_retriever_val = FullRetriever(documents=ds_val.documents)

        evaluations = {}
        # Compute the performance for the bm25 on the test dataset
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
                retriever=base_retriever_val.getReranker(  # type: ignore
                    scorer, valtopK, PowerAdaptativeBatcher()
                ),  # a retriever which use the splade model to score all the documents and then do the retrieve
                early_stop=early_stop,
                validation_interval=validation_interval,
                metrics={"RR@10": True, "AP": False, "nDCG@10": False},
            )

            learner = Learner(
                trainer=trainer,
                steps_per_epoch=steps_per_epoch,
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
                best = outputs.listeners["bestval"][
                    "RR@10"
                ]  # get the best trained model for the metrics RR@10
                retriever = create_retriever(
                    best
                )  # create a retriever by using the model

                evaluations[key].append(
                    evaluate(
                        token=token,
                        dataset=test,
                        retriever=retriever,
                        launcher=gpulauncher,
                    )
                )

        # generator a batchwise sampler which is an Iterator of ProductRecords()
        ibn_sampler = PairwiseInBatchNegativesSampler(sampler=train_sampler)
        scheduler = CosineWithWarmup(num_warmup_steps=num_warmup_steps)

        # Define the model and the flop loss for regularization
        # Model of class: DotDense(),
        # The parameters are the regularization coeff for the query and document
        spladev2, flops = spladeV2(3e-4, 1e-4)

        # Define the trainer for training the splade in a Batchwise trainer
        batchwise_trainer_flops = BatchwiseTrainer(
            batch_size=splade_batch_size,
            sampler=ibn_sampler,
            lossfn=SoftmaxCrossEntropy(),
            hooks=[flops],
        )

        # Train splade
        run(
            spladev2.tag("model", "splade-v2"),
            batchwise_trainer_flops,
            [
                ParameterOptimizer(
                    scheduler=scheduler, optimizer=AdamW(lr=1e-5, weight_decay=1e-2)
                )
            ],
            lambda scorer: sparse_retriever(scorer, documents),
            hooks=[setmeta(DistributedHook(models=[spladev2.encoder]), True)],
            launcher=gpulauncher_mem48,
        )

        # Train Dense

        # distilbert_encoder = TransformerEncoder(model_id=distilbert, trainable=True)
        # siamese = DotDense(
        #     query_encoder=distilbert_encoder.with_maxlength(30),
        #     encoder=distilbert_encoder.with_maxlength(200),
        # ).tag("model", "siamese")
        # adam_7 = [
        #     ParameterOptimizer(
        #         optimizer=Adam(lr=tag(7e-6)).tag("optim", "Adam"),
        #         scheduler=CosineWithWarmup(num_warmup_steps=num_warmup_steps),
        #     )
        # ]
        # batchwise_trainer = BatchwiseTrainer(
        #     batch_size=batch_size, sampler=ibn_sampler, lossfn=SoftmaxCrossEntropy()
        # )
        # run(siamese, batchwise_trainer, adam_7, faiss_retriever, launcher=gpulauncher2x)

        # Wait that experiments complete
        xp.wait()

        for key, dsevaluations in evaluations.items():
            print(f"=== {key}")
            for evaluation in dsevaluations:
                if evaluation.results.is_file():
                    print(
                        f"Results for {evaluation.__xpm__.tags()}\n{evaluation.results.read_text()}\n"
                    )


if __name__ == "__main__":
    cli()
