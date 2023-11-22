import click
from pathlib import Path
import os
from datamaestro import prepare_dataset
import logging
import multiprocessing

from experimaestro import experiment
from xpmir.evaluation import Evaluate
from xpmir.rankers.standard import BM25
from xpmir.interfaces.anserini import AnseriniRetriever, IndexCollection


logging.basicConfig(level=logging.INFO)
CPU_COUNT = multiprocessing.cpu_count()

# --- Defines the experiment


@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.option("--dataset", default="gov.nist.trec.adhoc.1")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, workdir, dataset, debug):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    bm25 = BM25()

    # Sets the working directory and the name of the xp
    with experiment(workdir, "bm25", port=port) as xp:
        # Index the collection
        xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])
        ds = prepare_dataset(dataset)

        documents = ds.documents
        index = IndexCollection(
            documents=documents,
            storePositions=True,
            storeDocvectors=True,
            storeContents=True,
            threads=CPU_COUNT,
        ).submit()

        # Search with BM25
        bm25_retriever = AnseriniRetriever(k=1500, index=index, model=bm25).tag(
            "model", "bm25"
        )

        bm25_eval = Evaluate(dataset=ds, retriever=bm25_retriever).submit()

    logging.info("BM25 results on TREC 1")
    logging.info(bm25_eval.results.read_text())


if __name__ == "__main__":
    cli()
