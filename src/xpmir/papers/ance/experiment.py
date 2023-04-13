import logging

from experimaestro import experiment
from experimaestro.launcherfinder import find_launcher
from xpmir.letor.batchers import PowerAdaptativeBatcher
from xpmir.letor.optim import (
    TensorboardService,
)
from xpmir.letor.samplers import NegativeSamplerListener
from xpmir.papers.cli import paper_command
from xpmir.rankers.standard import BM25
from xpmir.papers.results import PaperResults
from xpmir.papers.helpers.msmarco import (
    v1_tests,
    v1_validation_dataset,
    v1_passages,
)
from .configuration import ANCE
import xpmir.interfaces.anserini as anserini
from xpmir.index.faiss import FaissBuildListener, DynamicFaissIndex

logging.basicConfig(level=logging.INFO)


def run(
    xp: experiment, cfg: ANCE, tensorboard_service: TensorboardService
) -> PaperResults:
    """monoBERT model"""

    launcher_learner = find_launcher(cfg.ance.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_index = find_launcher(cfg.indexation.requirements)
    device = cfg.device  # noqa F401
    random = cfg.random  # noqa F401

    documents = v1_passages()  # noqa F401
    ds_val = v1_validation_dataset(cfg.validation)  # noqa F401

    tests = v1_tests()

    basemodel = BM25().tag("model", "bm25")

    # evaluate the baseline
    bm25_retriever = anserini.AnseriniRetriever(
        k=cfg.retrieval.k,
        index=anserini.index_builder(launcher=launcher_index),
        model=basemodel,
    )
    tests.evaluate_retriever(bm25_retriever, launcher_index)

    ance_trainer = ...  # noqa F401
    ance_model = ...  # noqa F401
    validation_listener = ...  # noqa F401
    faiss_listener = FaissBuildListener(  # noqa F401
        indexing_interval=cfg.ance.indexing_interval,
        indexbackedfaiss=DynamicFaissIndex(
            normalize=False,
            encoder=...,
            indexspec=cfg.ance.indexspec,
            batchsize=2048,
            batcher=PowerAdaptativeBatcher(),
            device=cfg.device,
            hooks=...,
        ),
    )
    sampler_listener = NegativeSamplerListener(...)  # noqa F401

    learner = ...

    outputs = learner.submit(launcher=launcher_learner)
    tensorboard_service.add(learner, learner.logpath)

    ance_final_index = ...  # noqa F401
    ance_retriever = ...

    tests.evaluate_retriever(
        ance_retriever,
        launcher_evaluate,
        model_id="ance-RR@10",
    )

    return PaperResults(
        models={"monobert-RR@10": outputs.listeners["bestval"]["RR@10"]},
        evaluations=tests,
        tb_logs={"monobert-RR@10": learner.logpath},
    )


@paper_command(schema=ANCE, package=__package__, tensorboard_service=True)
def cli(xp: experiment, cfg: ANCE, tensorboard_service: TensorboardService):
    return run(xp, cfg, tensorboard_service)
