# Information Retrieval for experimaestro


IR module for experimaestro

## Install

Install with

```
pip install experimaestro_ir
```

Specific extra dependencies can be used if you plan to use some
specific part of this module, e.g. for neural models

```
pip install experimaestro_ir[neural]
```

## Example


```python
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
from experimaestro_ir.neural.capreolus import DRMM, ModelLearn

# --- Defines the experiment


@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--port", type=int, default=12345, help="Port for monitoring")
@click.argument("workdir", type=Path)
@click.command()
def cli(port, workdir, debug):
    """Runs an experiment"""
    logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)

    from experimaestro_ir.anserini import IndexCollection, SearchCollection

    bm25 = BM25()

    # Sets the working directory and the name of the xp
    with experiment(workdir, "index", port=port) as xp:
        # Index the collection
        trec1 = prepare_dataset("gov.nist.trec.adhoc.1")
        trec2 = prepare_dataset("gov.nist.trec.adhoc.2")
        documents = trec1.documents
        documents.tag("dataset", "trec-1")
        index = IndexCollection(
            documents=documents,
            storePositions=True,
            storeDocvectors=True,
            storeTransformedDocs=True,
            threads=CPU_COUNT,
        ).submit()

        # Search with BM25
        bm25_search = (
            SearchCollection(index=index, topics=trec1.topics, model=bm25)
            .tag("model", "bm25")
            .submit()
        )
        bm25_eval = TrecEval(
            assessments=trec1.assessments, results=bm25_search
        ).submit()

        # Train with MatchZoo
        training = [prepare_dataset("gov.nist.trec.adhoc.2")]
        model = DRMM().tag("ranker", "match_pyramid")
        learnedmodel = ModelLearn(model=model, training=training).submit()

        # Re-order resutls with MatchZoo
        search = Reorder(results=bm25, topics=trec1.topics, model=learnedmodel).submit()
        eval = TrecEval(assessments=trec1.assessments, results=search).submit()


if __name__ == "__main__":
    cli()
```
