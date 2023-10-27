import logging
from functools import partial

import xpmir.letor.trainers.generative as generative
from experimaestro import experiment
from experimaestro.launcherfinder import find_launcher
from xpmir.neural.generative.probtab import ProbaTabIdentifierGenerator
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.learner import Learner
from xpmir.learning.optim import TensorboardService
from xpmir.letor.samplers import TripletBasedInBatchNegativeSampler
from xpmir.papers import configuration
from xpmir.papers.cli import paper_command
from xpmir.papers.helpers.msmarco import (
    v1_docpairs_sampler,
    v1_passages,
    v1_tests,
)
from xpmir.papers.monobert.configuration import Monobert
from xpmir.papers.monobert.experiment import get_retrievers
from xpmir.papers.results import PaperResults
from xpmir.rankers import RandomScorer

logging.basicConfig(level=logging.INFO)


# FIXME: shouldn't use the Monobert base configuration / confusing parameters
@configuration()
class T5GenerativeConfigurationProbaTab(Monobert):
    decoder_outdim: int = 4
    max_depth: int = 2
    base: str = "t5-base"
    """Identifier for the base model"""


def run(
    xp: experiment,
    cfg: T5GenerativeConfigurationProbaTab,
    tensorboard_service: TensorboardService,
) -> PaperResults:
    launcher_learner = find_launcher(cfg.monobert.requirements)
    launcher_preprocessing = find_launcher(cfg.preprocessing.requirements)
    device = cfg.device
    random = cfg.random

    documents = v1_passages()

    tests = v1_tests(cfg.dev_test_size)

    # Setup indices and validation/test base retrievers
    retrievers, model_based_retrievers = get_retrievers(cfg)
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

    proba_tab_model: ProbaTabIdentifierGenerator = ProbaTabIdentifierGenerator(
        hf_id=cfg.base, decoder_outdim=cfg.decoder_outdim, nb_docs=32
    )

    # define the trainer for monobert
    proba_tab_trainer = generative.GenerativeTrainer(
        loss=generative.PairwiseGenerativeRetrievalLoss(
            id_generator=proba_tab_model, max_depth=cfg.max_depth
        ),
        sampler=TripletBasedInBatchNegativeSampler(
            sampler=v1_docpairs_sampler(
                sample_rate=cfg.monobert.sample_rate,
                sample_max=cfg.monobert.sample_max,
                launcher=launcher_preprocessing,
            ),
            batch_size=16,
        ),
        batcher=PowerAdaptativeBatcher(),
        batch_size=cfg.monobert.optimization.batch_size,
    )

    # --- Learning

    # The learner trains the model
    learner = Learner(
        # Misc settings
        device=device,
        random=random,
        # How to train the model
        trainer=proba_tab_trainer,
        # The model to train (splade contains all the parameters)
        model=proba_tab_model,
        # Optimization settings
        steps_per_epoch=cfg.monobert.optimization.steps_per_epoch,
        optimizers=cfg.monobert.optimization.optimizer_splade,
        max_epochs=cfg.monobert.optimization.max_epochs,
        # The hook used for evaluation
        listeners=[],
        use_pretasks=True,
    )  # load from huggingface before learning

    # Submit job and link
    learner.submit(launcher=launcher_learner)
    tensorboard_service.add(learner, learner.logpath)


@paper_command(
    schema=T5GenerativeConfigurationProbaTab, folder=__file__, tensorboard_service=True
)
def cli(
    xp: experiment,
    cfg: T5GenerativeConfigurationProbaTab,
    tensorboard_service: TensorboardService,
):
    return run(xp, cfg, tensorboard_service)
