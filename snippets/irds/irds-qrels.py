from experimaestro import experiment
from datamaestro import prepare_dataset
from xpmir.measures import AP, nDCG
from xpmir.interfaces.anserini import IndexCollection, AnseriniRetriever
from xpmir.rankers.standard import BM25
from xpmir.evaluation import Evaluate
import os
import logging

logging.basicConfig(level=logging.INFO)
collection = prepare_dataset("irds.antique.train")

with experiment("workdir", "evaluate-bm25", port=12345) as xp:
    # Build the index
    xp.setenv("JAVA_HOME", os.environ["JAVA_HOME"])
    index = IndexCollection(documents=collection.documents).submit()

    bm25_retriever = AnseriniRetriever(k=1500, index=index, model=BM25())
    bm25_eval = Evaluate(
        dataset=collection, retriever=bm25_retriever, measures=[AP, nDCG @ 10]
    ).submit()

print("BM25 results")
print(bm25_eval.results.read_text())
