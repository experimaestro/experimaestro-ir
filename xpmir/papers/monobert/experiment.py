import io
import logging

import click
from xpmir.distributed import DistributedHook
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.schedulers import LinearWithWarmup
from xpmir.models import XPMIRHFHub
import xpmir.letor.trainers.pairwise as pairwise
from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from datamaestro_text.data.ir import AdhocDocuments
from xpmir.neural.cross import CrossScorer
from experimaestro import experiment, setmeta, tagspath
from experimaestro.launcherfinder import find_launcher
from experimaestro.utils import cleanupdir
from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluations, EvaluationsCollection
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor import Device, Random
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.letor.devices import CudaDevice
from xpmir.letor.optim import (
    AdamW,
    ParameterOptimizer,
    RegexParameterFilter,
    get_optimizers,
)
from xpmir.letor.samplers import TripletBasedSampler
from xpmir.measures import AP, RR, P, nDCG
from xpmir.papers.cli import paper_command
from xpmir.rankers import CollectionBasedRetrievers, RandomScorer, RetrieverHydrator
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import DualTransformerEncoder
from xpmir.utils.utils import find_java_home

logging.basicConfig(level=logging.INFO)


@click.option(
    "--upload-to-hub",
    required=False,
    type=str,
    default=None,
    help="Upload the model to Hugging Face Hub with the given identifier",
)
@paper_command(package=__package__)
def cli(debug, configuration, host, port, workdir, upload_to_hub, documentation, env):
    """monoBERT trained on MS-Marco

    Passage Re-ranking with BERT (Rodrigo Nogueira, Kyunghyun Cho). 2019.
    https://arxiv.org/abs/1901.04085
    """

    # Get launcher tags
    tags = configuration.launcher.tags.split(",") if configuration.launcher.tags else []

    # Number of batches per epoch (# samples = steps_per_epoch * batch_size)
    steps_per_epoch = configuration.learner.steps_per_epoch

    # Validation interval (in epochs)
    validation_interval = configuration.learner.validation_interval

    # How many document to re-rank for the monobert
    topK = configuration.retrieval.k

    # How many documents to use for cross-validation in monobert
    valtopK = configuration.retrieval.val_k

    # The numbers of the warmup steps during the training
    num_warmup_steps = configuration.learner.num_warmup_steps

    batch_size = configuration.learner.batch_size
    max_epochs = configuration.learner.max_epoch

    launcher_index = find_launcher(configuration.indexation.requirements)

    launcher_learner = find_launcher(configuration.learner.requirements, tags=tags)
    launcher_evaluate = find_launcher(configuration.evaluation.requirements, tags=tags)

    logging.info(
        f"Number of epochs {max_epochs}, validation interval {validation_interval}"
    )

    assert max_epochs % validation_interval == 0, (
        f"Number of epochs ({max_epochs}) is not a multiple "
        f"of validation interval ({validation_interval})"
    )

    # Sets the working directory and the name of the xp
    name = configuration.type

    with experiment(workdir, name, host=host, port=port) as xp:
        # Needed by Pyserini
        xp.setenv("JAVA_HOME", find_java_home())
        for key, value in env:
            xp.setenv(key, value)

        # Misc
        device = CudaDevice() if configuration.launcher.gpu else Device()
        random = Random(seed=0)
        basemodel = BM25().tag("model", "bm25")

        # create a random scorer as the most naive baseline
        random_scorer = RandomScorer(random=random).tag("reranker", "random")
        measures = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10]

        # Creates the directory with tensorboard data
        runs_path = xp.resultspath / "runs"
        cleanupdir(runs_path)
        runs_path.mkdir(exist_ok=True, parents=True)
        logging.info("You can monitor learning with:")
        logging.info("tensorboard --logdir=%s", runs_path)

        # Datasets: train, validation and test
        documents: AdhocDocuments = prepare_dataset("irds.msmarco-passage.documents")

        # for indexing
        # cars_documents: AdhocDocuments = prepare_dataset("irds.car.v1.5.documents")

        # Development datasets
        devsmall = prepare_dataset("irds.msmarco-passage.dev.small")
        dev = prepare_dataset("irds.msmarco-passage.dev")
        ds_val = RandomFold(
            dataset=dev,
            seed=123,
            fold=0,
            sizes=[configuration.learner.validation_size],
            exclude=devsmall.topics,
        ).submit()

        # Prepare the evaluation on MsMarco-v1 (dev) and TREC DL 2019/2020
        tests = EvaluationsCollection(
            msmarco_dev=Evaluations(devsmall, measures),
            trec2019=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2019"), measures
            ),
            trec2020=Evaluations(
                prepare_dataset("irds.msmarco-passage.trec-dl-2020"), measures
            ),
        )

        # Setup indices and validation/test base retrievers
        retrievers = CollectionBasedRetrievers()

        msmarco_index = IndexCollection(documents=documents).submit(
            launcher=launcher_index
        )
        retrievers.add(documents, index=msmarco_index)

        # cars_index = IndexCollection(
        #   documents=cars_documents,
        # ).submit(launcher=launcher_index)
        # retrievers.add(cars_documents, index=cars_index)

        val_retrievers = retrievers.factory(
            lambda documents, **kwargs: RetrieverHydrator(
                store=documents,
                retriever=AnseriniRetriever(**kwargs, k=valtopK, model=basemodel),
            )
        )
        test_retrievers = retrievers.factory(
            lambda documents, **kwargs: RetrieverHydrator(
                store=documents,
                retriever=AnseriniRetriever(**kwargs, k=topK, model=basemodel),
            )
        )

        # Search and evaluate with the base model
        tests.evaluate_retriever(test_retrievers, launcher_index)

        tests.evaluate_retriever(
            random_scorer.getRetrieverFactory(
                test_retrievers, batch_size, PowerAdaptativeBatcher()
            )
        )

        # Defines how we sample train examples
        # (using the shuffled pre-computed triplets from MS Marco)
        train_triples = prepare_dataset("irds.msmarco-passage.train.docpairs")
        triplesid = ShuffledTrainingTripletsLines(
            seed=123,
            data=train_triples,
        ).submit()
        train_sampler = TripletBasedSampler(source=triplesid, index=documents)

        # define the trainer for monobert
        monobert_trainer = pairwise.PairwiseTrainer(
            lossfn=pairwise.PointwiseCrossEntropyLoss(),
            sampler=train_sampler,
            batcher=PowerAdaptativeBatcher(),
            batch_size=batch_size,
        )

        monobert_scorer: CrossScorer = CrossScorer(
            encoder=DualTransformerEncoder(trainable=True, maxlen=512, dropout=0.1)
        ).tag("reranker", "monobert")

        # The validation listener will evaluate the full retriever
        # (1st stage + reranker) and keep the best performing model
        # on the validation set
        validation = ValidationListener(
            dataset=ds_val,
            retriever=monobert_scorer.getRetriever(
                val_retrievers(documents), batch_size, PowerAdaptativeBatcher()
            ),
            validation_interval=validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG": False},
        )

        # Setup the parameter optimizers
        scheduler = LinearWithWarmup(
            num_warmup_steps=num_warmup_steps,
            min_factor=configuration.learner.warmup_min_factor,
        )

        optimizers = [
            ParameterOptimizer(
                scheduler=scheduler,
                optimizer=AdamW(lr=configuration.learner.lr, eps=1e-6),
                filter=RegexParameterFilter(includes=[r"\.bias$", r"\.LayerNorm\."]),
            ),
            ParameterOptimizer(
                scheduler=scheduler,
                optimizer=AdamW(
                    lr=configuration.learner.lr, weight_decay=1e-2, eps=1e-6
                ),
            ),
        ]

        # The learner trains the model
        learner = Learner(
            # Misc settings
            device=device,
            random=random,
            # How to train the model
            trainer=monobert_trainer,
            # The model to train
            scorer=monobert_scorer,
            # Optimization settings
            steps_per_epoch=steps_per_epoch,
            optimizers=get_optimizers(optimizers),
            max_epochs=max_epochs,
            # The listeners (here, for validation)
            listeners={"bestval": validation},
            # The hook used for evaluation
            hooks=[setmeta(DistributedHook(models=[monobert_scorer]), True)],
        )

        # Submit job and link
        runs_path = runs_path or (experiment.current().resultspath / "runs")
        outputs = learner.submit(launcher=launcher_learner)
        (runs_path / tagspath(learner)).symlink_to(learner.logpath)

        # Evaluate the neural model
        for metric_name, monitored in validation.metrics.items():
            if monitored:
                best = outputs.listeners["bestval"][metric_name]
                tests.evaluate_retriever(
                    best.getRetrieverFactory(
                        test_retrievers,
                        batch_size,
                        PowerAdaptativeBatcher(),
                        device=device,
                    ),
                    launcher_evaluate,
                )

        # Waits that experiments complete
        xp.wait()

        if upload_to_hub:
            logging.info("Uploading to HuggingFace Hub")
            best = outputs.listeners["bestval"]["RR@10"]
            # TODO: add tags automatically + results
            out = io.StringIO()
            tests.output_results(out)
            documentation = f"""---
library_name: xpmir
---
{documentation}

{out.getvalue()}
"""
            # AutoModel.push_to_hf_hub(best.__unwrap__(),
            # repo_url=upload_to_hub, readme=documentation)
            XPMIRHFHub(best.__unwrap__(), readme=documentation).push_to_hub(
                repo_id=upload_to_hub
            )

        # Display metrics for each trained model
        tests.output_results()


if __name__ == "__main__":
    cli()
