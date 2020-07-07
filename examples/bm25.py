import click
from pathlib import Path
import os
from datamaestro import prepare_dataset
import logging
import multiprocessing


logging.basicConfig(level=logging.INFO)
CPU_COUNT = multiprocessing.cpu_count()


from experimaestro import experiment
from experimaestro_ir.evaluation import TrecEval
from experimaestro_ir.models import BM25
from experimaestro_ir.anserini import IndexCollection, SearchCollection

# --- Defines the experiment


@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, workdir, debug):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    bm25 = BM25()

    # Sets the working directory and the name of the xp
    with experiment(workdir, "index", port=port) as xp:
        # Index the collection
        xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])
        trec1 = prepare_dataset("gov.nist.trec.adhoc.1")

        documents = trec1.documents
        index = IndexCollection(
            documents=documents,
            storePositions=True,
            storeDocvectors=True,
            storeContents=True,
            threads=CPU_COUNT,
        ).submit()

        # Search with BM25
        bm25_search = (
            SearchCollection(index=index, topics=trec1.topics, model=bm25)
            .tag("model", "bm25")
            .submit()
        )
        bm25_eval = TrecEval(assessments=trec1.assessments, run=bm25_search).submit()


if __name__ == "__main__":
    cli()
