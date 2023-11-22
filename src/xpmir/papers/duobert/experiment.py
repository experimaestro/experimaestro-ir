# Implementation of the experiments in the paper Multi-Stage Document Ranking
# with BERT (Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, Jimmy Lin). 2019.
# https://arxiv.org/abs/1910.14424

# An imitation of examples/msmarco-reranking.py


from functools import partial
import logging
from experimaestro.launcherfinder import find_launcher
from xpmir.distributed import DistributedHook
from xpmir.learning.learner import Learner
from xpmir.letor.learner import ValidationListener
from xpmir.learning.optim import TensorboardService

import xpmir.letor.trainers.pairwise as pairwise
from xpmir.neural.cross import DuoCrossScorer
from experimaestro import experiment, setmeta
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.papers.cli import paper_command
from xpmir.papers.helpers.samplers import (
    prepare_collection,
    msmarco_v1_docpairs_sampler,
    msmarco_v1_tests,
    msmarco_v1_validation_dataset,
)
from xpmir.text.huggingface import DualDuoBertTransformerEncoder
from xpmir.papers.monobert.experiment import (
    get_retrievers,
    run as monobert_run,
)
from xpmir.papers.results import PaperResults
from .configuration import DuoBERT

logging.basicConfig(level=logging.INFO)


def run(xp: experiment, cfg: DuoBERT, tensorboard_service: TensorboardService):
    monobert_results = monobert_run(xp, cfg, tensorboard_service)

    launcher_learner = find_launcher(cfg.monobert.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)

    monobert_scorer = monobert_results.models["monobert-RR@10"]
    documents = prepare_collection("irds.msmarco-passage.documents")
    device = cfg.device
    random = cfg.random
    ds_val = msmarco_v1_validation_dataset(cfg.validation)
    tests = msmarco_v1_tests()

    # ------Start the code for the duobert

    # Define the trainer for the duobert
    duobert_trainer = pairwise.DuoPairwiseTrainer(
        lossfn=pairwise.PairwiseLossWithTarget().tag("loss", "duo_logp"),
        sampler=msmarco_v1_docpairs_sampler(),
        batcher=PowerAdaptativeBatcher(),
        batch_size=cfg.duobert.optimization.batch_size,
    )

    # The scorer(model) for the duobert
    duobert_scorer: DuoCrossScorer = DuoCrossScorer(
        encoder=DualDuoBertTransformerEncoder(trainable=True, dropout=0.1)
    ).tag("duo-reranker", "duobert")

    # Validation: we use monoBERT but only keep validation_top_k
    # results

    retrievers, model_based_retrievers = get_retrievers(cfg)

    monobert_val_retrievers = partial(
        model_based_retrievers,
        retrievers=partial(retrievers, k=cfg.duobert.base_validation_top_k),
        top_k=cfg.duobert.validation_top_k,
        scorer=monobert_scorer,
    )

    val_retriever = model_based_retrievers(
        documents, retrievers=monobert_val_retrievers, scorer=duobert_scorer
    )

    # The validation listener evaluates the full retriever
    # (retriever + reranker) and keep the best performing model
    # on the validation set
    validation = ValidationListener(
        id="bestval",
        dataset=ds_val,
        retriever=val_retriever,
        validation_interval=cfg.duobert.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG": False},
    )

    # The learner for the duobert.
    learner = Learner(
        # Misc settings
        device=device,
        random=random,
        # How to train the model
        trainer=duobert_trainer,
        # The model to train
        model=duobert_scorer,
        # Optimization settings
        steps_per_epoch=cfg.duobert.optimization.steps_per_epoch,
        optimizers=cfg.duobert.optimization.optimizer,
        max_epochs=cfg.duobert.optimization.max_epochs,
        # The listeners (here, for validation)
        listeners=[validation],
        # The hook used for evaluation
        hooks=[setmeta(DistributedHook(models=[duobert_scorer]), True)],
        use_fp16=True,
    )

    # Submit job and link
    outputs = learner.submit(launcher=launcher_learner)
    tensorboard_service.add(learner, learner.logpath)

    # Evaluate the neural model on test collections

    monobert_test_retrievers = partial(
        model_based_retrievers,
        retrievers=partial(retrievers, k=cfg.retrieval.base_k),
        top_k=cfg.retrieval.k,
        scorer=monobert_scorer,
    )
    test_retrievers = partial(
        model_based_retrievers,
        retrievers=monobert_test_retrievers,
        scorer=duobert_scorer,
    )

    for metric_name in validation.monitored():
        model = outputs.listeners["bestval"][metric_name]  # type: DuoCrossScorer
        tests.evaluate_retriever(
            partial(
                model_based_retrievers,
                scorer=model,
                retrievers=test_retrievers,
                device=device,
            ),
            launcher_evaluate,
            model_id=f"duobert-{metric_name}",
        )

    return PaperResults(
        models={"duobert-RR@10": outputs.listeners["bestval"]["RR@10"]},
        evaluations=tests,
        tb_logs={"duobert-RR@10": learner.logpath},
    )


@paper_command(package=__package__, schema=DuoBERT, tensorboard_service=True)
def cli(xp: experiment, cfg: DuoBERT, tensorboard_service: TensorboardService):
    return run(xp, cfg, tensorboard_service)
