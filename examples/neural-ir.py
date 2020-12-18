import logging
import os
from pathlib import Path
from datamaestro_text.data.ir import Adhoc
from xpmir.letor import Device, Random
from xpmir.letor.trainers.pointwise import PointwiseTrainer

from xpmir.rankers import TwoStageRetriever

from datamaestro import prepare_dataset
from experimaestro.click import click, forwardoption
from experimaestro import experiment, tag

# from onir.predictors.reranker import Device
# from onir.random import Random
from xpmir.letor.learner import Learner

# from onir.tasks.evaluate import Evaluate
# from onir.trainers.pointwise import PointwiseTrainer
from xpmir.rankers.standard import BM25
from xpmir.rankers.anserini import AnseriniRetriever, IndexCollection, SearchCollection
from xpmir.evaluation import Evaluate, TrecEval
from typing import List, Callable, NamedTuple

logging.basicConfig(level=logging.INFO)


# --- Experiment


@forwardoption.max_epoch(Learner)
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option(
    "--grad-acc-batch", type=int, default=64, help="Batch size for accumulating"
)
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
    models = []

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


def register(method, add):
    def m(**kwargs):
        return lambda info: add(info, method(info, **kwargs))

    m.__name__ = method.__name__
    m.__doc__ = method.__doc__
    return cli.command()(m)


# ---- Datasets


def dataset(method):
    return register(method, lambda info, ds: info.datasets.append(ds))


@dataset
def msmarco(info):
    """Use the MS Marco dataset"""
    logging.info("Adding MS Marco dataset")
    from xpmir.datasets.msmarco import MsmarcoDataset

    ds = MsmarcoDataset.prepare()
    return ds("train"), ds("dev"), ds("trec2019.test")


@dataset
def robust(info):
    """Use the TREC Robust dataset"""
    from xpmir.datasets.robust import fold

    # Return pairs topic/qrels
    pairs = [fold("trf1"), fold("vaf1"), fold("f1")]

    documents = prepare_dataset("gov.nist.trec.adhoc.robust.2004").documents

    return [
        Adhoc(topics=topics, assessments=qrels, documents=documents)
        for topics, qrels in pairs
    ]


# ---- Vocabulary


def vocab(method):
    return register(method, lambda info, vocab: setattr(info, "vocab", vocab))


@vocab
def glove(info):
    from xpmir.vocab.wordvec_vocab import WordvecUnkVocab

    wordembs = prepare_dataset("edu.stanford.glove.6b.50")
    return WordvecUnkVocab(data=wordembs, random=info.random)


@click.option(
    "--trainable", is_flag=True, help="Make the BERT encoder parameters trainable"
)
@vocab
def bertencoder(info, trainable):
    import xpmir.vocab.bert_vocab as bv

    return bv.BertVocab(train=trainable)


# ---- Models


def model(method):
    return register(method, lambda info, model: info.models.append(model))


@model
def drmm(info):
    """Use the DRMM model"""
    from xpmir.rankers.neural.drmm import Drmm

    assert info.vocab is not None, "No embeddings are defined yet for DRMM"
    return Drmm(vocab=info.vocab).tag("model", "drmm")


@model
def vanilla_transformer(info):
    """Use the Vanilla BERT model"""
    from xpmir.rankers.neural.vanilla_transformer import VanillaTransformer

    return VanillaTransformer(vocab=info.vocab).tag("model", "vanilla-transformer")


# --- Run the experiment


@cli.resultcallback()
def process(
    processors, debug, gpu, port, workdir, max_epoch, batch_size, grad_acc_batch
):
    """Runs an experiment"""
    logging.info("Running pipeline")

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    info = Information()
    info.device = device = Device(gpu=gpu)

    # Sets the working directory and the name of the xp
    with experiment(workdir, "neural-ir", port=port) as xpm:
        # Misc settings
        assert (
            "JAVA_HOME" in os.environ
        ), "JAVA_HOME should be defined (to call anserini)"
        xpm.setenv("JAVA_HOME", os.environ["JAVA_HOME"])

        # Prepare the embeddings
        info.device = device

        for processor in processors:
            processor(info)

        assert info.datasets, "No dataset was selected"
        assert info.models, "No model was selected"

        basemodel = BM25()

        for train, val, test in info.datasets:

            train_index, val_index, test_index = [
                info.index(c.documents) for c in (train, val, test)
            ]

            # Search and evaluate with BM25
            bm25_retriever = AnseriniRetriever(index=test_index, model=basemodel).tag(
                "model", "bm25"
            )
            bm25_eval = Evaluate(dataset=test, retriever=bm25_retriever).submit()

            # Train and evaluate with each model
            # for model in info.models:
            #     # Train with OpenNIR DRMM model
            #     # predictor = Reranker(device=device, batch_size=batch_size)
            #     trainer = PointwiseTrainer(device=device, grad_acc_batch=grad_acc_batch)
            #     reranker = TwoStageRetriever(base=basemodel, reranker=model)
            #     learner = Learner(
            #         sampler=sampler, random=random, model=model,
            #         ranker=reranker,
            #         train_dataset=train, val_dataset=val, max_epoch=tag(max_epoch)
            #     )
            #     model = learner.submit()

            #     # Evaluate the neural model
            #     evaluate = Evaluate(dataset=test, ranker=reranker).submit()

        xpm.wait()

        print(f"Results for BM25\n{bm25_eval.results.read_text()}\n")
        # print(f"Results for DRMM\n{evaluate.results.read_text()}\n")


if __name__ == "__main__":
    cli()
