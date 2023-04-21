from functools import partial
import logging

from xpmir.distributed import DistributedHook
from xpmir.letor.learner import Learner, ValidationListener
import xpmir.letor.trainers.pairwise as pairwise
from xpmir.neural.cross import CrossScorer
from experimaestro import experiment, setmeta
from experimaestro.launcherfinder import find_launcher
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.letor.optim import (
    TensorboardService,
)
from xpmir.papers.cli import paper_command
from xpmir.rankers.standard import BM25
from xpmir.text.huggingface import DualTransformerEncoder
from xpmir.papers.results import PaperResults
from xpmir.utils.functools import cache
from xpmir.papers.helpers.msmarco import (
    v1_docpairs_sampler,
    v1_tests,
    v1_validation_dataset,
    v1_passages,
)
from .configuration import Monobert
import xpmir.interfaces.anserini as anserini
from xpmir.rankers import scorer_retriever, RandomScorer

logging.basicConfig(level=logging.INFO)


@cache
def get_retrievers(cfg: Monobert):
    """Returns retrievers


    :param cfg: The configuration
    :return: A tuple composed of (1) a retriever factory based on the base model
        (BM25) and (2)
    """
    launcher_index = cfg.indexation.launcher

    base_model = BM25().tag("model", "bm25")

    retrievers = partial(
        anserini.retriever,
        anserini.index_builder(launcher=launcher_index),
        model=base_model,
    )  #: Anserini based retrievers

    model_based_retrievers = partial(
        scorer_retriever,
        batch_size=cfg.retrieval.batch_size,
        batcher=PowerAdaptativeBatcher(),
        device=cfg.device,
    )  #: Model-based retrievers

    return retrievers, model_based_retrievers


def run(
    xp: experiment, cfg: Monobert, tensorboard_service: TensorboardService
) -> PaperResults:
    """monoBERT model"""

    launcher_learner = find_launcher(cfg.monobert.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    device = cfg.device
    random = cfg.random

    documents = v1_passages()
    ds_val = v1_validation_dataset(cfg.validation)

    tests = v1_tests()

    # Setup indices and validation/test base retrievers
    retrievers, model_based_retrievers = get_retrievers(cfg)

    test_retrievers = partial(retrievers, k=cfg.retrieval.k)  #: Test retrievers

    # Search and evaluate with a random re-ranker
    random_scorer = RandomScorer(random=random).tag("scorer", "random")
    tests.evaluate_retriever(
        partial(
            model_based_retrievers,
            retrievers=test_retrievers,
            scorer=random_scorer,
            device=None,
        )
    )

    # Search and evaluate with the base model
    tests.evaluate_retriever(test_retrievers, cfg.indexation.launcher)

    # Define the different launchers
    val_retrievers = partial(retrievers, k=cfg.monobert.validation_top_k)

    # define the trainer for monobert
    monobert_trainer = pairwise.PairwiseTrainer(
        lossfn=pairwise.PointwiseCrossEntropyLoss(),
        sampler=v1_docpairs_sampler(),
        batcher=PowerAdaptativeBatcher(),
        batch_size=cfg.monobert.optimization.batch_size,
    )

    monobert_scorer: CrossScorer = CrossScorer(
        encoder=DualTransformerEncoder(
            model_id="bert-base-uncased", trainable=True, maxlen=512, dropout=0.1
        )
    ).tag("scorer", "monobert")

    # The validation listener evaluates the full retriever
    # (retriever + scorer) and keep the best performing model
    # on the validation set
    validation = ValidationListener(
        id="bestval",
        dataset=ds_val,
        retriever=model_based_retrievers(
            documents,
            retrievers=val_retrievers,
            scorer=monobert_scorer,
            device=device,
        ),
        validation_interval=cfg.monobert.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG": False},
    )

    # The learner trains the model
    learner = Learner(
        # Misc settings
        device=device,
        random=random,
        # How to train the model
        trainer=monobert_trainer,
        # The model to train
        model=monobert_scorer,
        # Optimization settings
        steps_per_epoch=cfg.monobert.optimization.steps_per_epoch,
        optimizers=cfg.monobert.optimization.optimizer,
        max_epochs=cfg.monobert.optimization.max_epochs,
        # The listeners (here, for validation)
        listeners=[validation],
        # The hook used for evaluation
        hooks=[setmeta(DistributedHook(models=[monobert_scorer]), True)],
    )

    # Submit job and link
    outputs = learner.submit(launcher=launcher_learner)
    tensorboard_service.add(learner, learner.logpath)

    # Evaluate the neural model on test collections
    for metric_name in validation.monitored():
        model = outputs.listeners["bestval"][metric_name]  # type: CrossScorer
        tests.evaluate_retriever(
            partial(
                model_based_retrievers,
                scorer=model,
                retrievers=test_retrievers,
                device=device,
            ),
            launcher_evaluate,
            model_id=f"monobert-{metric_name}",
        )

    return PaperResults(
        models={"monobert-RR@10": outputs.listeners["bestval"]["RR@10"]},
        evaluations=tests,
        tb_logs={"monobert-RR@10": learner.logpath},
    )


@paper_command(schema=Monobert, package=__package__, tensorboard_service=True)
def cli(xp: experiment, cfg: Monobert, tensorboard_service: TensorboardService):
    return run(xp, cfg, tensorboard_service)
