# Implementation of the experiments in the paper
# Multi-Stage Document Ranking with BERT (Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, Jimmy Lin). 2019.
# https://arxiv.org/abs/1910.14424

# An imitation of examples/msmarco-reranking.py

# flake8: noqa: T201

import logging
from pathlib import Path
from omegaconf import OmegaConf
from xpmir.distributed import DistributedHook
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.schedulers import LinearWithWarmup

import xpmir.letor.trainers.pairwise as pairwise
from datamaestro import prepare_dataset
from datamaestro_text.data.ir import AdhocDocuments
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from xpmir.neural.cross import CrossScorer, DuoCrossScorer
from experimaestro import experiment, setmeta, tagspath
from experimaestro.click import click
from experimaestro.launcherfinder import cpu, find_launcher
from experimaestro.launcherfinder.specs import duration
from experimaestro.utils import cleanupdir
from xpmir.configuration import omegaconf_argument
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
from xpmir.pipelines.reranking import RerankingPipeline
from xpmir.rankers import CollectionBasedRetrievers, RandomScorer, RetrieverHydrator
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import DualDuoBertTransformerEncoder, DualTransformerEncoder
from xpmir.utils.utils import find_java_home

logging.basicConfig(level=logging.INFO)


@paper_command(package=__package__)
def cli(debug, configuration, host, port, workdir, env):
    """Runs an experiment"""

    # Get launcher tags
    tags = configuration.launcher.tags.split(",") if configuration.launcher.tags else []

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    # Number of topics in the validation set
    VAL_SIZE = configuration.learner.validation_size

    # Number of batches per epoch (# samples = STEPS_PER_EPOCH * batch_size)
    steps_per_epoch = configuration.learner.steps_per_epoch

    # Validation interval (in epochs)
    validation_interval = configuration.learner.validation_interval

    # How many document to re-rank for the monobert
    topK1 = configuration.retriever_mono.k
    # How many documents to use for cross-validation in monobert
    valtopK1 = configuration.retriever_mono.val_k

    # How many document to pass from the monobert to duobert
    topK2 = configuration.retriever_duo.k
    # How many document to use for cross-validation in duobert
    valtopK2 = configuration.retriever_duo.val_k

    # The numbers of the warmup steps during the training
    num_warmup_steps = configuration.learner.num_warmup_steps

    # Our default launcher for light tasks
    gpu_launcher_index = find_launcher(configuration.indexation.requirements)

    batch_size = configuration.learner.batch_size
    max_epochs = configuration.learner.max_epoch

    # we assigne a gpu, if not, a cpu
    launcher_index = find_launcher(configuration.indexation.requirements)

    launcher_learner = find_launcher(configuration.learner.requirements, tags=tags)
    launcher_evaluate = find_launcher(configuration.evaluation.requirements, tags=tags)

    logging.info(
        f"Number of epochs {max_epochs}, validation interval {validation_interval}"
    )

    assert (
        max_epochs % validation_interval == 0
    ), f"Number of epochs ({max_epochs}) is not a multiple of validation interval ({validation_interval})"

    # Sets the working directory and the name of the xp
    name = configuration.type

    with experiment(workdir, name, host=host, port=port) as xp:
        # Needed by Pyserini
        xp.setenv("JAVA_HOME", find_java_home())
        for key, value in env:
            xp.setenv(key, value)

        # Misc
        device = CudaDevice() if configuration.Launcher.gpu else Device()
        random = Random(seed=0)
        basemodel = BM25()

        # create a random scorer as the most naive baseline
        random_scorer = RandomScorer(random=random).tag("model", "random")
        measures = [AP, P @ 20, nDCG, nDCG @ 10, nDCG @ 20, RR, RR @ 10]

        # Creates the directory with tensorboard data
        runs_path = xp.resultspath / "runs"
        cleanupdir(runs_path)
        runs_path.mkdir(exist_ok=True, parents=True)
        logging.info("You can monitor learning with:")
        logging.info("tensorboard --logdir=%s", runs_path)

        # Datasets: train, validation and test
        # for indexing --> msmarcos
        documents: AdhocDocuments = prepare_dataset("irds.msmarco-passage.documents")

        # Not using the trec_car for now
        # for indexing --> trec_car
        # cars_documents = prepare_dataset(
        #     "irds.car.v1.5.documents"
        # )

        # Development datasets
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
        )

        # Setup indices and validation/test base retrievers
        retrievers = CollectionBasedRetrievers()

        # Build the MS Marco index and definition of first stage rankers
        msmarco_index = IndexCollection(documents=documents).submit(launcher_index)

        retrievers.add(documents, index=msmarco_index)

        # Also can add the index for the trec_car, but not for now
        # cars_index = IndexCollection(
        #   documents=cars_documents, storeContents=True
        # ).submit(launcher=launcher_index)
        # retrievers.add(cars_documents, index=cars_index)

        # define the base retriever for validation and testing on the msmarcos
        # index respectively
        val_retrievers = retrievers.factory(
            lambda documents, **kwargs: RetrieverHydrator(
                store=documents,
                retriever=AnseriniRetriever(**kwargs, k=valtopK1, model=basemodel),
            )
        )
        test_retrievers = retrievers.factory(
            lambda documents, **kwargs: RetrieverHydrator(
                store=documents,
                retriever=AnseriniRetriever(**kwargs, k=topK1, model=basemodel),
            )
        )

        # Search and evaluate with the base model
        tests.evaluate_retriever(test_retrievers, launcher_index)

        # Search and evaluate with the bm25 model
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
            lossfn=pairwise.PointwiseCrossEntropyLoss().tag("loss", "pce"),
            sampler=train_sampler,
            batcher=PowerAdaptativeBatcher(),
            batch_size=batch_size,
        )

        scheduler = LinearWithWarmup(
            num_warmup_steps=num_warmup_steps,
            min_factor=configuration.Learner.warmup_min_factor,
        )

        monobert_scorer: CrossScorer = CrossScorer(
            encoder=DualTransformerEncoder(trainable=True, maxlen=512, dropout=0.1)
        ).tag("reranker", "monobert")

        monobert_validation = ValidationListener(
            dataset=ds_val,
            retriever=monobert_scorer.getRetriever(
                val_retrievers(documents), batch_size, PowerAdaptativeBatcher()
            ),
            validation_interval=validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG": False},
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
        mono_learner = Learner(
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
            listeners={"bestval": monobert_validation},
            # The hook used for evaluation
            hooks=[setmeta(DistributedHook(models=[monobert_scorer]), True)],
        )

        # Submit the job of learning the monobert and link
        mono_runs_path = runs_path or (experiment.current().resultspath / "runs")
        outputs = mono_learner.submit(launcher=launcher_learner)
        (mono_runs_path / tagspath(mono_learner)).symlink_to(mono_learner.logpath)

        # Evaluate the monobert model
        for metric_name, monitored in monobert_validation.metrics.items():
            if monitored:
                best = outputs.listeners["bestval"][metric_name]
                tests.evaluate_retriever(
                    best.getRetrieverFactory(
                        test_retrievers, batch_size, PowerAdaptativeBatcher()
                    ),
                    launcher_evaluate,
                )

        # ------Start the code for the duobert

        # Get the trained monobert model for reranking
        best_mono_scorer = outputs.listeners["bestval"]["RR@10"]

        # Define the trainer for the duobert
        duobert_trainer = pairwise.DuoPairwiseTrainer(
            lossfn=pairwise.DuoLogProbaLoss().tag("loss", "duo_proba"),
            sampler=train_sampler,
            batcher=PowerAdaptativeBatcher(),
            batch_size=batch_size,
        )

        # define the retriever factories for the duobert for test and validation(only for msmarcos now)
        duobert_val_retrievers = best_mono_scorer.getRetrieverFactory(
            val_retrievers,
            batch_size,
            PowerAdaptativeBatcher(),
            top_k=valtopK2,
            device=device,
        )
        duobert_test_retrievers = best_mono_scorer.getRetrieverFactory(
            test_retrievers,
            batch_size,
            PowerAdaptativeBatcher(),
            top_k=topK2,
            device=device,
        )

        # The scorer(model) for the duobert
        duobert_scorer: DuoCrossScorer = DuoCrossScorer(
            encoder=DualDuoBertTransformerEncoder(trainable=True, dropout=0.1)
        ).tag("duo-reranker", "duobert")

        duobert_validation = ValidationListener(
            dataset=ds_val,
            retriever=duobert_scorer.getRetriever(
                duobert_val_retrievers(documents),
                batch_size,
                PowerAdaptativeBatcher(),
                device=device,
            ),
            validation_interval=validation_interval,
            metrics={"RR@10": True, "AP": False, "nDCG": False},
        )

        # The learner for the duobert.
        duobert_learner = Learner(
            # Misc settings
            device=device,
            random=random,
            # How to train the model
            trainer=duobert_trainer,
            # The model to train
            scorer=duobert_scorer,
            # Optimization settings
            steps_per_epoch=steps_per_epoch,
            optimizers=get_optimizers(optimizers),
            max_epochs=max_epochs,
            # The listeners (here, for validation)
            listeners={"bestval": duobert_validation},
            # The hook used for evaluation
            hooks=[setmeta(DistributedHook(models=[duobert_scorer]), True)],
        )

        # Submit job and link
        duo_runs_path = runs_path or (experiment.current().resultspath / "runs")
        outputs = duobert_learner.submit(launcher=launcher_learner)
        (duo_runs_path / tagspath(duobert_learner)).symlink_to(duobert_learner.logpath)

        # Evaluate the duobert model
        for metric_name, monitored in monobert_validation.metrics.items():
            if monitored:
                best = outputs.listeners["bestval"][metric_name]
                tests.evaluate_retriever(
                    best.getRetriever(
                        duobert_test_retrievers(documents),
                        batch_size,
                        PowerAdaptativeBatcher(),
                    ),
                    launcher_evaluate,
                )

        # Waits that experiments complete
        xp.wait()

        # ---  End of the experiment
        # Display metrics for each trained model
        tests.output_results()


if __name__ == "__main__":
    cli()
