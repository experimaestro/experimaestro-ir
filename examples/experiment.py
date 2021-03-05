import logging
from pathlib import Path
from functools import lru_cache

from datamaestro import prepare_dataset
from datamaestro_text.transforms.ir import ShuffledTrainingTripletsLines
from experimaestro import experiment, tag, tagspath
from experimaestro.click import click, forwardoption
from experimaestro.utils import cleanupdir
from xpmir.datasets.adapters import RandomFold
from xpmir.evaluation import Evaluate
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection
from xpmir.letor import Device, Random
from xpmir.letor.learner import Learner, Validation
from xpmir.letor.optim import Adam
from xpmir.letor.samplers import ModelBasedSampler, Sampler, TripletBasedSampler
from xpmir.letor.trainers import Trainer
from xpmir.letor.trainers.pairwise import PairwiseTrainer
from xpmir.neural.drmm import Drmm
from xpmir.neural.colbert import Colbert
from xpmir.rankers import RandomScorer, TwoStageRetriever
from xpmir.rankers.standard import BM25
from xpmir.vocab.huggingface import TransformerVocab
from xpmir.vocab.wordvec_vocab import WordvecUnkVocab

logging.basicConfig(level=logging.INFO)


class Information:
    def __init__(self):
        self._indexes = {}

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


# --- Experiment
@forwardoption.max_epoch(Learner)
@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--gpu", is_flag=True, help="Use GPU")
@click.option(
    "--batch-size", type=int, default=256, help="Batch size (validation and test)"
)
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.command()
def cli(debug, gpu, port, workdir, max_epoch, batch_size):
    """Runs an experiment"""

    BATCHES_PER_EPOCH = 32

    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
    device = Device(gpu=gpu)
    info = Information()

    # Sets the working directory and the name of the xp
    with experiment(workdir, "msmarco", port=port) as xp:
        # Train / validation / test
        train_triples = prepare_dataset("com.microsoft.msmarco.passage.train.idtriples")
        dev = prepare_dataset("com.microsoft.msmarco.passage.dev")
        test = prepare_dataset("com.microsoft.msmarco.passage.trec2019.test")
        index = info.index(train_triples.documents)
        test_index = index

        # Base models
        random = Random(seed=0)
        basemodel = BM25()
        random_scorer = RandomScorer(random=random).tag("model", "random")

        # Get a random subset of 500 queries for validation
        ds_val = RandomFold(dataset=dev, seed=102, size=500).submit()

        triplesid = ShuffledTrainingTripletsLines(
            seed=123,
            data=prepare_dataset("com.microsoft.msmarco.passage.train.idtriples"),
        ).submit()
        train_sampler = TripletBasedSampler(source=triplesid, index=index)

        # Retrieve the top 1000
        topK = 1000
        # 1000 documents used for cross-validation
        valtopK = 100

        # @lru_cache
        def get_reranker(index, scorer, topk=topK):
            base_retriever = AnseriniRetriever(k=topk, index=index, model=basemodel)
            return TwoStageRetriever(
                retriever=base_retriever, scorer=scorer, batchsize=batch_size
            )

        # Search and evaluate with BM25
        bm25_retriever = AnseriniRetriever(
            k=topK, index=test_index, model=basemodel
        ).tag("model", "bm25")
        bm25_eval = Evaluate(dataset=test, retriever=bm25_retriever).submit()

        # Performance of random
        random_eval = Evaluate(
            dataset=test, retriever=get_reranker(test_index, random_scorer)
        ).submit()

        wordembs = prepare_dataset("edu.stanford.glove.6b.50")
        glove = WordvecUnkVocab(data=wordembs, random=random)

        @lru_cache
        def trainer(lr=1e-3, grad_acc_batch=0):
            return PairwiseTrainer(
                optimizer=Adam(lr=lr),
                device=device,
                batches_per_epoch=BATCHES_PER_EPOCH,
                sampler=train_sampler,
                grad_acc_batch=grad_acc_batch,
                batch_size=batch_size,
            )

        # Train and evaluate with each model
        evaluations = []
        runspath = xp.resultspath / "runs"
        cleanupdir(runspath)
        runspath.mkdir(exist_ok=True, parents=True)

        token = xp.token("main", 1)

        def run(scorer, trainer: Trainer):
            validation = Validation(
                dataset=ds_val, retriever=get_reranker(index, scorer, valtopK)
            )

            learner = Learner(
                trainer=trainer,
                random=random,
                scorer=scorer,
                validation_interval=16,
                max_epoch=tag(max_epoch),
                validation=validation,
            )
            model = token(1, learner).submit()
            (runspath / tagspath(model)).symlink_to(model.logpath)

            # Evaluate the neural model
            evaluations.append(
                token(
                    1, Evaluate(dataset=test, retriever=get_reranker(index, model))
                ).submit()
            )

        # DRMM
        drmm = Drmm(vocab=glove, index=index).tag("model", "drmm")
        run(drmm, trainer(lr=tag(1e-3)))

        # We use micro-batches of size 8 for BERT-based models
        colbert = Colbert(vocab=TransformerVocab(trainable=True), dlen=512).tag(
            "model", "colbert"
        )
        run(colbert, trainer(lr=tag(1e-3), grad_acc_batch=4))

        # Wait that experiments complete
        xp.wait()

        print(f"Results for BM25\n{bm25_eval.results.read_text()}\n")
        print(f"Results for random\n{random_eval.results.read_text()}\n")
        for evaluate in evaluations:
            print(f"Results for {evaluate.tags()}\n{evaluate.results.read_text()}\n")


if __name__ == "__main__":
    cli()
