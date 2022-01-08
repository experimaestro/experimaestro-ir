import functools
import logging
from pathlib import Path
from typing import Any, Optional
from datamaestro_text.data.ir import Adhoc
from xpmir.letor import Device, Random
from xpmir.letor.samplers import Sampler, ModelBasedSampler, TripletBasedSampler
from xpmir.letor.trainers import Trainer
from xpmir.letor.trainers.pointwise import PointwiseTrainer
from xpmir.letor.trainers.pairwise import PairwiseTrainer
from xpmir.neural.interaction import InteractionScorer
from xpmir.neural.interaction.drmm import Drmm

from xpmir.rankers import RandomScorer, TwoStageRetriever

from datamaestro import prepare_dataset
from experimaestro.click import click, forwardoption
from experimaestro import experiment, tag

from xpmir.letor.learner import Learner, Validation

from xpmir.rankers.standard import BM25
from xpmir.interfaces.anserini import (
    AnseriniRetriever,
    IndexCollection,
)
from xpmir.evaluation import Evaluate
from xpmir.text.huggingface import TransformerVocab

logging.basicConfig(level=logging.INFO)


# --- Experiment


@forwardoption.max_epoch(Learner)
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option(
    "--batch-size", type=int, default=64, help="Batch size (validation and test)"
)
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.group(chain=True, invoke_without_command=False)
def cli(**kwargs):
    pass


class Information:
    vocab = None
    device = None
    _random = None
    _indexes = {}

    datasets = []
    scorers = []
    basemodel = BM25()

    train_sampler: Optional[Sampler] = None
    dev: Any = None
    test: Any = None
    trainer: Trainer

    def index(self, ds):
        """Returns the anserini index"""
        index = self._indexes.get(ds.id)
        if index is None:
            import multiprocessing

            CPU_COUNT = multiprocessing.cpu_count()

            index = IndexCollection(
                documents=ds,
                storePositions=True,
                storeDocvectors=True,
                storeContents=True,
                threads=CPU_COUNT,
            ).submit()
            self._indexes[ds.id] = index
        return index

    @property
    def random(self):
        if not self._random:
            self._random = Random()

        return self._random


def _register(method, add):
    def m(**kwargs):
        def _m(info: Information):
            r = method(info, **kwargs)
            if add is not None:
                add(info, r)

        return _m

    m.__name__ = method.__name__
    m.__doc__ = method.__doc__
    return cli.command()(m)


def register(method):
    return _register(method, None)


# --- Trainers


@click.option("--batch-size", type=int, default=64, help="Batch size for training")
@click.option(
    "--grad-acc-batch", type=int, default=0, help="Batch size for accumulating"
)
@register
def pointwise(info: Information, batch_size, grad_acc_batch):
    info.trainer = PointwiseTrainer(
        device=info.device,
        sampler=info.train_sampler,
        grad_acc_batch=grad_acc_batch,
        batch_size=batch_size,
    )


@click.option("--batch-size", type=int, default=64, help="Batch size for training")
@click.option(
    "--grad-acc-batch", type=int, default=0, help="Batch size for accumulating"
)
@register
def pairwise(info: Information, batch_size, grad_acc_batch):
    info.trainer = PairwiseTrainer(
        device=info.device,
        sampler=info.train_sampler,
        grad_acc_batch=grad_acc_batch,
        batch_size=batch_size,
    )


# ---- Datasets


def dataset(method):
    return _register(method, None)


@functools.lru_cache()
def _msmarco_docs():
    return prepare_dataset("com.microsoft.msmarco.passage.collection")


def _msmarco(part: str):
    return Adhoc(
        documents=_msmarco_docs(),
        topics=prepare_dataset(f"com.microsoft.msmarco.passage.{part}.queries"),
        assessments=prepare_dataset(f"com.microsoft.msmarco.passage.{part}.qrels"),
    )


@click.option("--top-k", type=int, default=1000, help="Top-k for model-based sampler")
@dataset
def msmarco(info: Information, top_k: int):
    """Use the MS Marco dataset"""
    logging.info("Adding MS Marco dataset")

    info.train_sampler = ModelBasedSampler(
        retriever=AnseriniRetriever(
            k=top_k, index=info.index(_msmarco_docs()), model=info.basemodel
        ),
        dataset=_msmarco("train"),
    )

    info.dev = _msmarco("dev")


@dataset
def msmarco_train_triplets(info: Information):
    """Use MS-Marco triplets"""
    info.train_sampler = TripletBasedSampler(
        source=prepare_dataset("com.microsoft.msmarco.passage.train.idtriples"),
        index=info.index(_msmarco_docs()),
    )


@dataset
def msmarco_test2019(info: Information):
    info.test = _msmarco("trec2019.test")


@click.option("--top-k", type=int, default=1000, help="Top-k for model-based sampler")
@dataset
def robust(info: Information, top_k: int):
    """Use the TREC Robust dataset"""
    from xpmir.datasets.robust import fold

    # Return pairs topic/qrels
    documents = prepare_dataset("gov.nist.trec.adhoc.robust.2004").documents

    def get(p: str):
        topics, qrels = fold(p)
        return Adhoc(topics=topics, assessments=qrels, documents=documents)

    info.train_sampler = ModelBasedSampler(
        retriever=AnseriniRetriever(
            k=top_k, index=info.index(documents), model=info.basemodel
        ),
        dataset=get("trf1"),
    )

    info.dev = get("trf1")
    info.test = get("f1")


# ---- Vocabulary


def vocab(method):
    return _register(method, lambda info, vocab: setattr(info, "vocab", vocab))


@vocab
def glove(info):
    from xpmir.text.wordvec_vocab import WordvecUnkVocab

    wordembs = prepare_dataset("edu.stanford.glove.6b.50")
    return WordvecUnkVocab(data=wordembs, random=info.random)


@forwardoption.model_id(TransformerVocab)
@click.option(
    "--trainable", is_flag=True, help="Make the BERT encoder parameters trainable"
)
@vocab
def bertencoder(info, trainable, model_id):
    import xpmir.text.huggingface as bv

    return bv.IndependentTransformerVocab(trainable=trainable, model_id=model_id)


# ---- scorers


def model(method):
    return _register(method, lambda info, model: info.scorers.append(model))


@forwardoption.dlen(InteractionScorer)
@forwardoption.qlen(InteractionScorer)
@forwardoption.combine(Drmm)
@model
def drmm(info, dlen, qlen, combine):
    """Use the DRMM model"""
    assert info.vocab is not None, "No embeddings are defined yet for DRMM"
    return Drmm(vocab=info.vocab, dlen=dlen, qlen=qlen, combine=combine).tag(
        "model", "drmm"
    )


@model
def vanilla_transformer(info):
    """Use the Vanilla BERT model"""
    from xpmir.neural.vanilla_transformer import VanillaTransformer

    return VanillaTransformer(vocab=info.vocab).tag("model", "vanilla-transformer")


# --- Run the experiment


@cli.resultcallback()
def process(processors, debug, gpu, port, workdir, max_epoch, batch_size):
    """Runs an experiment"""
    logging.info("Running pipeline")

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    info = Information()
    info.device = device = Device(gpu=gpu)

    # Sets the working directory and the name of the xp
    with experiment(workdir, "neural-ir", port=port) as xpm:
        # Misc settings__xpm__
        # Prepare the embeddings
        info.device = device

        for processor in processors:
            processor(info)

        assert info.trainer, "No trainer was selected"
        assert info.train_sampler, "No train sampler was selected"
        assert info.dev, "No dev dataset was selected"
        assert info.test, "No test dataset was selected"

        random_scorer = RandomScorer(random=info.random).tag("model", "random")

        # Retrieve the top 1000
        topK = 1000
        # 1000 documents used for cross-validation
        valtopK = 100

        def get_retriever(index, scorer, topk=topK):
            base_retriever = AnseriniRetriever(
                k=topk, index=index, model=info.basemodel
            )
            return TwoStageRetriever(
                retriever=base_retriever, scorer=scorer, batchsize=batch_size
            )

        val_index, test_index = [info.index(c.documents) for c in (info.dev, info.test)]

        # Search and evaluate with BM25
        bm25_retriever = AnseriniRetriever(
            k=topK, index=test_index, model=info.basemodel
        ).tag("model", "bm25")
        bm25_eval = Evaluate(dataset=info.test, retriever=bm25_retriever).submit()

        # Performance of random
        random_eval = Evaluate(
            dataset=info.test, retriever=get_retriever(test_index, random_scorer)
        ).submit()

        # Train and evaluate with each model
        for scorer in info.scorers:
            # Train with OpenNIR DRMM model
            # predictor = Reranker(device=device, batch_size=batch_size)

            trainer = info.trainer
            validation = Validation(
                dataset=info.dev, retriever=get_retriever(val_index, scorer, valtopK)
            )

            learner = Learner(
                trainer=trainer,
                random=info.random,
                scorer=scorer,
                max_epoch=tag(max_epoch),
                validation=validation,
            )
            model = learner.submit()

            # Evaluate the neural model
            evaluate = Evaluate(
                dataset=info.test, retriever=get_retriever(test_index, model)
            ).submit()

        xpm.wait()

        print(f"===")
        print(f"Results for BM25\n{bm25_eval.results.read_text()}\n")
        print(f"Results for DRMM\n{evaluate.results.read_text()}\n")
        print(f"Results for random\n{random_eval.results.read_text()}\n")


if __name__ == "__main__":
    cli()
