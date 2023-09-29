import logging
from functools import partial

import xpmir.letor.trainers.generative as generative
from experimaestro import experiment, setmeta
from experimaestro.launcherfinder import find_launcher
from xpmir.neural.generative import GenerativeRetrievalScorer
from xpmir.neural.generative.hf import T5IdentifierGenerator, LoadFromT5
from xpmir.distributed import DistributedHook
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.learner import Learner
from xpmir.learning.optim import TensorboardService
from xpmir.letor.learner import ValidationListener
from xpmir.papers import configuration
from xpmir.papers.cli import paper_command
from xpmir.papers.helpers.msmarco import (
    v1_docpairs_sampler,
    v1_passages,
    v1_tests,
    v1_validation_dataset,
)
from xpmir.papers.monobert.configuration import Monobert
from xpmir.papers.monobert.experiment import get_retrievers
from xpmir.papers.results import PaperResults
from xpmir.rankers import RandomScorer

logging.basicConfig(level=logging.INFO)


# FIXME: shouldn't use the Monobert base configuration / confusing parameters
@configuration()
class T5GenerativeConfiguration(Monobert):
    base: str = "t5-base"
    """Identifier for the base model"""


def run(
    xp: experiment,
    cfg: T5GenerativeConfiguration,
    tensorboard_service: TensorboardService,
) -> PaperResults:
    launcher_learner = find_launcher(cfg.monobert.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
    device = cfg.device
    random = cfg.random

    documents = v1_passages()
    ds_val = v1_validation_dataset(cfg.validation, launcher=launcher_preprocessing)

    tests = v1_tests(cfg.dev_test_size)

    # Setup indices and validation/test base retrievers
    retrievers, model_based_retrievers = get_retrievers(cfg)
    val_retrievers = partial(
        retrievers, store=documents, k=cfg.monobert.validation_top_k
    )
    test_retrievers = partial(
        retrievers, store=documents, k=cfg.retrieval.k
    )  #: Test retrievers

    # Search and evaluate with a random re-ranker
    random_scorer = RandomScorer(random=random).tag("scorer", "random")
    tests.evaluate_retriever(
        partial(
            model_based_retrievers,
            retrievers=test_retrievers,
            scorer=random_scorer,
            device=None,
        ),
        launcher=launcher_preprocessing,
    )

    # Search and evaluate with the base model
    tests.evaluate_retriever(test_retrievers, cfg.indexation.launcher)

    t5_model: T5IdentifierGenerator = T5IdentifierGenerator(hf_id="t5-base").tag(
        "scorer", "t5"
    )  # not initialized from huggingface yet

    t5_scorer: GenerativeRetrievalScorer = GenerativeRetrievalScorer(
        id_generator=t5_model
    )

    # define the trainer for monobert
    t5_trainer = generative.GenerativeTrainer(
        loss=generative.PairwiseGenerativeRetrievalLoss(id_generator=t5_model),
        sampler=v1_docpairs_sampler(
            sample_rate=cfg.monobert.sample_rate,
            sample_max=cfg.monobert.sample_max,
            launcher=launcher_preprocessing,
        ),
        batcher=PowerAdaptativeBatcher(),
        batch_size=cfg.monobert.optimization.batch_size,
    )

    # The validation listener evaluates the full retriever
    # on the validation set
    # (retriever + scorer) and keep the best performing model
    validation = ValidationListener(
        id="t5_generative",
        dataset=ds_val,
        retriever=model_based_retrievers(
            documents,
            retrievers=val_retrievers,
            scorer=t5_scorer,
            device=device,
        ),
        validation_interval=cfg.monobert.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG": False},
    )

    # --- Learning

    # The learner trains the model
    learner = Learner(
        # Misc settings
        device=device,
        random=random,
        # How to train the model
        trainer=t5_trainer,
        # The model to train (splade contains all the parameters)
        model=t5_model,
        # Optimization settings
        steps_per_epoch=cfg.monobert.optimization.steps_per_epoch,
        optimizers=cfg.monobert.optimization.optimizer,
        max_epochs=cfg.monobert.optimization.max_epochs,
        # The listeners (here, for validation)
        listeners=[validation],
        # The hook used for evaluation
        hooks=[setmeta(DistributedHook(models=[t5_model]), True)],
    ).add_pretasks(
        LoadFromT5(model=t5_model)
    )  # load from huggingface before learning

    # Submit job and link
    outputs = learner.submit(launcher=launcher_learner)
    tensorboard_service.add(learner, learner.logpath)

    # Evaluate the neural model on test collections
    for metric_name in validation.monitored():
        trained: T5IdentifierGenerator = outputs.listeners[validation.id][
            metric_name
        ]  # returns the model from the learner

        # FIXME: Get the model from trained and transform it to a scorer
        trained_scorer = GenerativeRetrievalScorer(
            id_generator=t5_model.add_pretasks_from(trained)
        )

        tests.evaluate_retriever(
            partial(
                model_based_retrievers,
                scorer=trained_scorer,
                retrievers=test_retrievers,
                device=device,
            ),
            launcher_evaluate,
            model_id=f"monobert-{metric_name}",
        )

    return PaperResults(
        models={"t5d-RR@10": outputs.listeners[validation.id]["RR@10"]},
        evaluations=tests,
        tb_logs={"t5d-RR@10": learner.logpath},
    )


@paper_command(
    schema=T5GenerativeConfiguration, folder=__file__, tensorboard_service=True
)
def cli(
    xp: experiment,
    cfg: T5GenerativeConfiguration,
    tensorboard_service: TensorboardService,
):
    return run(xp, cfg, tensorboard_service)
