import logging
import os
from pathlib import Path
from datamaestro_text.data.ir import Adhoc
from xpmir.letor import Device, Random
from xpmir.letor.samplers import ModelBasedSampler
from xpmir.letor.trainers.pointwise import PointwiseTrainer
from xpmir.neural import InteractionScorer
from xpmir.neural.drmm import Drmm

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
from xpmir.vocab.huggingface import TransformerVocab

logging.basicConfig(level=logging.INFO)


# --- Experiment


@forwardoption.max_epoch(Learner)
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option(
    "--grad-acc-batch", type=int, default=0, help="Batch size for accumulating"
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
    scorers = []

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

    documents = prepare_dataset("com.microsoft.msmarco.passage.collection")

    l = []
    for p in ["train", "dev", "trec2019.test"]:
        l.append(
            Adhoc(
                documents=documents,
                topics=prepare_dataset(f"com.microsoft.msmarco.passage.{p}.queries"),
                assessments=prepare_dataset(f"com.microsoft.msmarco.passage.{p}.qrels"),
            )
        )
    return l


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


@forwardoption.model_id(TransformerVocab)
@click.option(
    "--trainable", is_flag=True, help="Make the BERT encoder parameters trainable"
)
@vocab
def bertencoder(info, trainable, model_id):
    import xpmir.vocab.huggingface as bv

    return bv.IndependentTransformerVocab(trainable=trainable, model_id=model_id)


# ---- scorers


def model(method):
    return register(method, lambda info, model: info.scorers.append(model))


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
        assert info.scorers, "No model was selected"

        basemodel = BM25()
        random_scorer = RandomScorer(random=info.random).tag("model", "random")

        # Retrieve the top 1000
        topK = 1000
        valtopK = 100

        def get_retriever(index, scorer, topk=topK):
            base_retriever = AnseriniRetriever(k=topk, index=index, model=basemodel)
            return TwoStageRetriever(
                retriever=base_retriever, scorer=scorer, batchsize=batch_size
            )

        for train, val, test in info.datasets:

            train_index, val_index, test_index = [
                info.index(c.documents) for c in (train, val, test)
            ]

            # Search and evaluate with BM25
            bm25_retriever = AnseriniRetriever(
                k=topK, index=test_index, model=basemodel
            ).tag("model", "bm25")
            bm25_eval = Evaluate(dataset=test, retriever=bm25_retriever).submit()
            random_eval = Evaluate(
                dataset=test, retriever=get_retriever(test_index, random_scorer)
            ).submit()

            # Train and evaluate with each model
            for scorer in info.scorers:
                # Train with OpenNIR DRMM model
                # predictor = Reranker(device=device, batch_size=batch_size)

                scorer.index = train_index
                sampler = ModelBasedSampler(
                    retriever=AnseriniRetriever(
                        k=topK, index=train_index, model=basemodel
                    ),
                    dataset=train,
                )
                trainer = PointwiseTrainer(
                    device=device,
                    sampler=sampler,
                    grad_acc_batch=grad_acc_batch,
                    batch_size=batch_size,
                )
                validation = Validation(
                    dataset=val, retriever=get_retriever(val_index, scorer, valtopK)
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
                    dataset=test, retriever=get_retriever(test_index, model)
                ).submit()

            xpm.wait()

            print(f"===")
            print(f"Results for BM25\n{bm25_eval.results.read_text()}\n")
            print(f"Results for DRMM\n{evaluate.results.read_text()}\n")
            print(f"Results for random\n{random_eval.results.read_text()}\n")


if __name__ == "__main__":
    cli()
