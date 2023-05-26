import logging

from experimaestro import experiment, setmeta
from experimaestro.launcherfinder import find_launcher
from xpmir.learning.batchers import PowerAdaptativeBatcher
from xpmir.learning.optim import TensorboardService
from xpmir.learning.learner import Learner
from xpmir.letor.samplers import PairwiseModelBasedSampler
from xpmir.papers.cli import paper_command
from xpmir.rankers.standard import BM25
from xpmir.papers.results import PaperResults
from xpmir.papers.helpers.msmarco import (
    v1_tests,
    v1_validation_dataset,
    v1_passages,
    v1_docpairs_sampler,
    v1_train_judged,
)
from .configuration import ANCE
import xpmir.interfaces.anserini as anserini
from xpmir.index.faiss import (
    DynamicFaissIndex,
    FaissRetriever,
    IndexBackedFaiss,
    InitFaissIndexBuildingHook,
)
from xpmir.datasets.adapters import RetrieverBasedCollection
import xpmir.letor.trainers.pairwise as pairwise
from xpmir.text.huggingface import TransformerEncoder
from xpmir.neural.dual import DotDense
from xpmir.letor.learner import (
    ValidationListener,
    NegativeSamplerListener,
    FaissBuildListener,
)
from xpmir.distributed import DistributedHook
from xpmir.rankers.full import FullRetriever

logging.basicConfig(level=logging.INFO)


def run(
    xp: experiment, cfg: ANCE, tensorboard_service: TensorboardService
) -> PaperResults:
    """monoBERT model"""

    basemodel = BM25().tag("model", "bm25")

    launcher_learner_warmup = find_launcher(cfg.ance_warmup.requirements)
    launcher_learner = find_launcher(cfg.ance.requirements)
    launcher_evaluate = find_launcher(cfg.retrieval.requirements)
    launcher_index = find_launcher(cfg.indexation.requirements)
    launcher_training_index = find_launcher(cfg.indexation.training_requirements)

    device = cfg.device
    random = cfg.random

    documents = v1_passages()
    ds_val_all = v1_validation_dataset(cfg.validation)

    # documents_train = RetrieverBasedCollection(
    #     dataset=v1_train_judged(),
    #     retrievers=[
    #         anserini.AnseriniRetriever(
    #             k=cfg.retrieval.trainTopK,
    #             index=anserini.index_builder()(documents=documents),
    #             model=basemodel,
    #         ),
    #     ],
    # ).submit(launcher=launcher_index)

    ds_val = RetrieverBasedCollection(
        dataset=ds_val_all,
        retrievers=[
            anserini.AnseriniRetriever(
                k=cfg.retrieval.retTopK,
                index=anserini.index_builder()(documents=documents),
                model=basemodel,
            ),
        ],
    ).submit(launcher=launcher_index)

    # Base retrievers for validation
    # It retrieve all the document of the collection with score 0 to avoid
    # scoring all the documents in the validation stage
    base_retriever_full = FullRetriever(documents=ds_val.documents)

    tests = v1_tests()

    # evaluate the baseline
    bm25_retriever = anserini.AnseriniRetriever(
        k=cfg.retrieval.topK,
        index=anserini.index_builder(launcher=launcher_index)(documents),
        model=basemodel,
    )
    tests.evaluate_retriever(bm25_retriever, launcher_index)

    encoder = TransformerEncoder(maxlen=512, model_id="roberta-base", trainable=True)

    ance_model = DotDense(encoder=encoder)

    validation_listener_warmup = ValidationListener(
        id="bestval_w",
        dataset=ds_val,
        retriever=ance_model.getRetriever(
            retriever=base_retriever_full,
            batch_size=cfg.retrieval.batch_size_full_retriever,
            batcher=PowerAdaptativeBatcher(),
            device=device,
        ),
        early_stop=cfg.ance_warmup.early_stop,
        validation_interval=cfg.ance_warmup.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG@10": False},
    )

    # better separate the warmup learning and mined training
    ance_warmup_trainer = pairwise.PairwiseTrainer(
        lossfn=pairwise.PointwiseCrossEntropyLoss(),
        sampler=v1_docpairs_sampler(),
        batcher=PowerAdaptativeBatcher(),
        batch_size=cfg.ance_warmup.optimization.batch_size,
    )

    warmup_learner = Learner(
        # Misc settings
        random=random,
        device=device,
        # How to train the model
        trainer=ance_warmup_trainer,
        # the model to be trained
        model=ance_model.tag("model", "ance"),
        # Optimization settings
        optimizers=cfg.ance_warmup.optimization.optimizer,
        steps_per_epoch=cfg.ance_warmup.optimization.steps_per_epoch,
        use_fp16=True,
        max_epochs=cfg.ance_warmup.optimization.max_epochs,
        # the listener for the validation
        listeners=[validation_listener_warmup],
        # the hooks
        hooks=[setmeta(DistributedHook(models=[encoder]), True)],
    )

    warmup_outputs = warmup_learner.submit(launcher=launcher_learner_warmup)
    tensorboard_service.add(warmup_learner, warmup_learner.logpath)

    warmup_model = warmup_outputs.listeners[validation_listener_warmup.id]["RR@10"]
    warmup_encoder = warmup_model.encoder

    # A faiss index which could be updated during the training
    dynamic_faiss = DynamicFaissIndex(
        documents=documents,  # number of documents may be reduced
        normalize=False,
        encoder=warmup_encoder,
        indexspec=cfg.indexation.indexspec,
        batchsize=2048,
        batcher=PowerAdaptativeBatcher(),
        device=device,
        hooks=[setmeta(DistributedHook(models=[warmup_encoder]), True)],
    )

    validation_listener = ValidationListener(
        id="bestval",
        dataset=ds_val,
        retriever=warmup_model.getRetriever(
            retriever=base_retriever_full,
            batch_size=cfg.retrieval.batch_size_full_retriever,
            batcher=PowerAdaptativeBatcher(),
            device=device,
        ),
        early_stop=cfg.ance.early_stop,
        validation_interval=cfg.ance.validation_interval,
        metrics={"RR@10": True, "AP": False, "nDCG@10": False},
        store_last_checkpoint=True,
    )

    faiss_listener = FaissBuildListener(
        id="faissbuilder",
        indexing_interval=cfg.ance.indexing_interval,
        indexbackedfaiss=dynamic_faiss,
    )

    sampler_listener = NegativeSamplerListener(
        id="negativebuilder", sampling_interval=cfg.ance.sampling_interval
    )

    # We warm up the ance model with the bm25 samplers and the swap to the
    # model_based_sampler
    modelbasedsampler = PairwiseModelBasedSampler(
        dataset=v1_train_judged(),
        retriever=FaissRetriever(
            encoder=warmup_encoder,
            index=dynamic_faiss,
            topk=cfg.retrieval.negative_sampler_topk,
            store=documents,
        ),
        batch_size=1024,
        max_query=cfg.retrieval.max_query,
    )

    ance_trainer = pairwise.PairwiseTrainer(
        lossfn=pairwise.PointwiseCrossEntropyLoss(),
        sampler=modelbasedsampler,
        batcher=PowerAdaptativeBatcher(),
        batch_size=cfg.ance.optimization.batch_size,
    )

    learner = Learner(
        # Misc settings
        random=random,
        device=device,
        # How to train the model
        trainer=ance_trainer,
        # the model to be trained
        model=warmup_model,
        # Optimization settings
        optimizers=cfg.ance.optimization.optimizer,
        steps_per_epoch=cfg.ance.optimization.steps_per_epoch,
        use_fp16=True,
        max_epochs=cfg.ance.optimization.max_epochs,
        # the listener for the validation
        listeners=[validation_listener, faiss_listener, sampler_listener],
        # the hooks
        hooks=[
            setmeta(DistributedHook(models=[warmup_encoder]), True),
            InitFaissIndexBuildingHook(indexbackedfaiss=dynamic_faiss),
        ],
    )

    outputs = learner.submit(launcher=launcher_learner)
    tensorboard_service.add(learner, learner.logpath)

    # get the trained model
    trained_model = outputs.listeners[validation_listener.id]["last_checkpoint"]

    ance_final_index = IndexBackedFaiss(
        normalize=False,
        documents=documents,
        encoder=trained_model.encoder,
        indexspec=cfg.indexation.indexspec,
        batchsize=2048,
        batcher=PowerAdaptativeBatcher(),
        device=device,
        hooks=[setmeta(DistributedHook(models=[trained_model.encoder]), True)],
    ).submit(launcher=launcher_training_index)

    ance_retriever = FaissRetriever(
        encoder=trained_model.encoder,
        index=ance_final_index,
        topk=cfg.retrieval.topK,
    )

    tests.evaluate_retriever(
        ance_retriever,
        launcher_evaluate,
        model_id="ance-RR@10",
    )

    return PaperResults(
        models={"ance-RR@10": outputs.listeners[validation_listener.id]["RR@10"]},
        evaluations=tests,
        tb_logs={"ance-RR@10": learner.logpath},
    )


@paper_command(schema=ANCE, package=__package__, tensorboard_service=True)
def cli(xp: experiment, cfg: ANCE, tensorboard_service: TensorboardService):
    return run(xp, cfg, tensorboard_service)
