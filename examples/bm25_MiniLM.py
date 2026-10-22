import logging
from pathlib import Path
from datamaestro import prepare_dataset
from experimaestro import experiment

from xpmir.rankers.standard import BM25
from xpmir.index.bow import BOWSparseRetrieverIndexBuilder, BOWRetriever
from xpmir.neural.sentence_transformers import STCrossEncoder
from xpmir.rankers import TwoStageRetriever
from xpmir.evaluation import Evaluate

logging.basicConfig(level=logging.INFO)

# Define experiment plan
with experiment(Path("./experiments/demo"), "nano_fever_bm25_minilm") as xp:
    # 1. Load benchmark dataset from Datamaestro
    dataset = prepare_dataset("co.huggingface.nano-beir.fever")

    # 2. Index documents (This experimaestro Task writes de BoW index)
    index = BOWSparseRetrieverIndexBuilder.C(documents=dataset.documents).submit()

    # 3. Define First stage BM25 retriever
    first_stage = BOWRetriever.C(
        index=index,
        model=BM25.C(),
        topk=100,
    ).tag("first_stage", "bm25")

    # 4. Define neural cross-encoder re-ranker
    scorer = STCrossEncoder.C(model_id="microsoft/MiniLM-L12-H384-uncased")

    # 4. Compose two-stage retriever: rerank top 50 with Cross-Encoder
    retriever = TwoStageRetriever.C(
        retriever=first_stage,
        scorer=scorer,
        top_k=50,
        batchsize=32,
    ).tag("model", "bm25_minilm_reranker")

    # 5. Submit evaluation task (computes nDCG@20, RR, AP via ir-measures)
    eval_task = Evaluate.C(dataset=dataset, retriever=retriever).submit()

print(f"Results written to: {eval_task.results}")
with open(eval_task.results) as f:
    metrics = f.read()
    print(f'Evaluation Metrics for "microsoft/MiniLM-L12-H384-uncased":\n{metrics}')
