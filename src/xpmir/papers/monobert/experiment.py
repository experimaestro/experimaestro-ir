import logging

from xpmir.distributed import DistributedHook
from xpmir.letor.learner import Learner, ValidationListener
from xpmir.letor.schedulers import LinearWithWarmup
import xpmir.letor.trainers.pairwise as pairwise
from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from datamaestro_text.data.ir import AdhocDocuments, Adhoc
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
from xpmir.papers.cli import UploadToHub, paper_command
from xpmir.rankers import CollectionBasedRetrievers, RandomScorer, RetrieverHydrator
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import DualTransformerEncoder
from xpmir.utils.utils import find_java_home
from .configuration import Monobert

logging.basicConfig(level=logging.INFO)


@paper_command(schema=Monobert, package=__package__)
def cli(xp: experiment, cfg: Monobert, upload_to_hub: UploadToHub):
    """monoBERT trained on MS-Marco

    Passage Re-ranking with BERT (Rodrigo Nogueira, Kyunghyun Cho). 2019.
    https://arxiv.org/abs/1901.04085
    """

    # Define the different launchers
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_learner = find_launcher(cfg.learner.requirements)
    launcher_evaluate = find_launcher(cfg.evaluation.requirements)

    # Sets the working directory and the name of the xp
    # Needed by Pyserini
    xp.setenv("JAVA_HOME", find_java_home())

    # Misc
    device = CudaDevice() if cfg.gpu else Device()
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
    devsmall: Adhoc = prepare_dataset("irds.msmarco-passage.dev.small")
    dev: Adhoc = prepare_dataset("irds.msmarco-passage.dev")

    # Sample the dev set to create a validation set
    ds_val = RandomFold(
        dataset=dev,
        seed=123,
        fold=0,
        sizes=[cfg.learner.validation_size],
        exclude=devsmall.topics,
    ).submit()

    # Prepares the test collections evaluation
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

    msmarco_index = IndexCollection(documents=documents).submit(launcher=launcher_index)
    retrievers.add(documents, index=msmarco_index)

    val_retrievers = retrievers.factory(
        lambda documents, **kwargs: RetrieverHydrator(
            store=documents,
            retriever=AnseriniRetriever(
                **kwargs, k=cfg.retrieval.val_k, model=basemodel
            ),
        )
    )
    test_retrievers = retrievers.factory(
        lambda documents, **kwargs: RetrieverHydrator(
            store=documents,
            retriever=AnseriniRetriever(**kwargs, k=cfg.retrieval.k, model=basemodel),
        )
    )

    # Search and evaluate with the base model
    tests.evaluate_retriever(test_retrievers, launcher_index)

    # Search and evaluate with a random reranker
    tests.evaluate_retriever(
        random_scorer.getRetrieverFactory(
            test_retrievers, cfg.retrieval.batch_size, PowerAdaptativeBatcher()
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
        batch_size=cfg.learner.batch_size,
    )

    monobert_scorer: CrossScorer = CrossScorer(
        encoder=DualTransformerEncoder(
            model_id="bert-base-uncased", trainable=True, maxlen=512, dropout=0.1
        )
    ).tag("reranker", "monobert")

    # The validation listener will evaluate the full retriever
    # (1st stage + reranker) and keep the best performing model
    # on the validation set
    validation = ValidationListener(
        dataset=ds_val,
        retriever=monobert_scorer.getRetriever(
            val_retrievers(documents),
            cfg.retrieval.batch_size,
            PowerAdaptativeBatcher(),
        ),
        validation_interval=cfg.learner.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG": False},
    )

    # Setup the parameter optimizers
    scheduler = LinearWithWarmup(
        num_warmup_steps=cfg.learner.num_warmup_steps,
        min_factor=cfg.learner.warmup_min_factor,
    )

    optimizers = [
        ParameterOptimizer(
            scheduler=scheduler,
            optimizer=AdamW(lr=cfg.learner.lr, eps=1e-6),
            filter=RegexParameterFilter(includes=[r"\.bias$", r"\.LayerNorm\."]),
        ),
        ParameterOptimizer(
            scheduler=scheduler,
            optimizer=AdamW(lr=cfg.learner.lr, weight_decay=1e-2, eps=1e-6),
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
        steps_per_epoch=cfg.learner.steps_per_epoch,
        optimizers=get_optimizers(optimizers),
        max_epochs=cfg.learner.max_epochs,
        # The listeners (here, for validation)
        listeners={"bestval": validation},
        # The hook used for evaluation
        hooks=[setmeta(DistributedHook(models=[monobert_scorer]), True)],
    )

    # Submit job and link
    runs_path = runs_path or (experiment.current().resultspath / "runs")
    outputs = learner.submit(launcher=launcher_learner)
    (runs_path / tagspath(learner)).symlink_to(learner.logpath)

    # Evaluate the neural model on test collections
    for metric_name in validation.monitored():
        model = outputs.listeners["bestval"][metric_name]
        tests.evaluate_retriever(
            model.getRetrieverFactory(
                test_retrievers,
                cfg.learner.batch_size,
                PowerAdaptativeBatcher(),
                device=device,
            ),
            launcher_evaluate,
            model_id=f"monobert-{metric_name}",
        )

    # Waits that experiments complete
    xp.wait()

    # Upload to HUB if requested
    upload_to_hub.send_scorer(
        {"monobert-RR@10": outputs.listeners["bestval"]["RR@10"]}, evaluations=tests
    )

    # Display metrics for each trained model
    tests.output_results()


if __name__ == "__main__":
    cli()
