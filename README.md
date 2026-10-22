<div align="center">

# Experimaestro IR (`xpmir`)

**A modular, reproducible Information Retrieval framework for modern neural ranking, dense/sparse retrieval, and distillation.**

[![PyPI version](https://badge.fury.io/py/experimaestro-ir.svg)](https://badge.fury.io/py/experimaestro-ir)
[![Documentation Status](https://readthedocs.org/projects/experimaestro-ir/badge/?version=latest)](https://experimaestro-ir.readthedocs.io/en/latest/?badge=latest)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

</div>

---

## 📌 Overview

**`experimaestro-ir` (`xpmir`)** is the Information Retrieval extension for [Experimaestro](https://experimaestro-python.readthedocs.io/). It provides a composable, type-safe library of IR primitives for designing, training, and benchmarking neural and classical retrieval pipelines.

Built on top of [xpm-torch](https://github.com/experimaestro/xpm-torch) and [datamaestro](https://datamaestro-ir.readthedocs.io/), `xpmir` bridges research experiments and distributed production pipelines:
- **Reproducible by Design**: Every dataset preparation, index, training run, and evaluation metric is a declared Experimaestro task with content-based caching and dependency tracking.
- **Modern Neural IR Architectures**: Native support for Cross-Encoders, Learned Sparse Retrieval (SPLADE, SEISMIC), Late-Interaction / Multi-Vector (ColBERT, PLAID), and Dense Bi-Encoders.
- **Advanced LETOR & Distillation**: Pointwise, Pairwise (MarginMSE, RankNet), and Listwise (InfoNCE) training objectives, hard negative sampling, and teacher-student distillation.
- **Scalable Evaluation**: Automated benchmarking against MS MARCO, TREC-DL, BEIR, NanoBEIR, and Robust04 using `ir-measures`, featuring multi-GPU distributed evaluation.

> 💡 **Ecosystem & Projects:** Explore research papers, models, and benchmarks built on top of XPMIR in the [xpmir GitHub organization](https://github.com/xpmir).

---

## 📦 Installation

`xpmir` supports both [`uv`](https://docs.astral.sh/uv/) and `pip`.

### Using `uv` (Recommended)
```bash
# Core package
uv add experimaestro-ir

# With Neural IR (PyTorch, Hugging Face Transformers, Sentence Transformers)
uv add "experimaestro-ir[neural]"

# With Classical Search (Anserini / Pyserini)
uv add "experimaestro-ir[anserini]"

# With fast PLAID multi-vector indexing
uv add "experimaestro-ir[fast-plaid]"
```

### Using `pip`
```bash
pip install "experimaestro-ir[neural]"
```

---

## ⚡ Quickstart: Evaluating a Two-Stage Pipeline

The following minimal example sets up a two-stage re-ranking experiment (first-stage retrieval + neural cross-encoder scorer) and evaluates it on a benchmark dataset:

```python
import logging
from pathlib import Path
from datamaestro import prepare_dataset
from experimaestro import experiment

from xpmir.neural.cross import CrossEncoderScorer
from xpmir.rankers.retriever import TwoStageRetriever
from xpmir.evaluation import Evaluate

logging.basicConfig(level=logging.INFO)

# Define experiment plan
with experiment(Path("./experiments/demo"), "two_stage_reranking") as xp:
    # 1. Load benchmark dataset (e.g. TREC Deep Learning or NanoBEIR)
    dataset = prepare_dataset("irds.msmarco-passage.trec-dl-2019")

    # 2. Define first-stage retriever (e.g., BM25 or SPLADE)
    # first_stage = ...

    # 3. Define neural re-ranker
    scorer = CrossEncoderScorer(
        model_id="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    # 4. Compose two-stage retriever (retrieve top 1000, rerank top 100)
    retriever = TwoStageRetriever(
        retriever=first_stage,
        k=1000,
        scorer=scorer,
        rerank_k=100,
    ).tag("model", "minilm-reranker")

    # 5. Submit evaluation task (computes nDCG@10, MRR@10, MAP via ir-measures)
    eval_task = Evaluate(dataset=dataset, retriever=retriever).submit()

print(f"Results path: {eval_task.results.path}")
```

---

## 🧩 Core Architecture & Modules

```
xpmir/
├── neural/         # Neural IR architectures (CrossEncoder, ColBERT, SPLADE, BiEncoders)
├── rankers/        # Scorers, TwoStageRetriever, MonolithicRetriever, Fusion
├── letor/          # Learning-to-Rank: trainers, batchers, samplers (MarginMSE, InfoNCE)
├── index/          # Index interfaces: dense (FAISS), sparse inverted index, SEISMIC
├── evaluation.py   # Benchmark evaluation pipelines powered by ir-measures
├── interfaces/     # External integrations: Anserini, Hugging Face, SentenceTransformers
└── datasets/       # Dataset adapters for datamaestro-ir and ir-datasets
```

| Component | Capabilities |
|---|---|
| **[Scorers & Encoders](https://experimaestro-ir.readthedocs.io/en/latest/neural.html)** | Cross-Encoders, Dual-Encoders, ColBERT late interaction, SPLADE sparse activations, DRMM. |
| **[Samplers & Training](https://experimaestro-ir.readthedocs.io/en/latest/letor/index.html)** | Dynamic batching, hard negative mining (BM25, SPLADE, ColBERT negatives), pairwise/listwise distillation. |
| **[Retrievers & Indices](https://experimaestro-ir.readthedocs.io/en/latest/retrieval.html)** | `TwoStageRetriever`, BM25 (Anserini), FAISS dense index, Rust-accelerated sparse inverted index, PLAID. |
| **[Evaluation & Benchmarks](https://experimaestro-ir.readthedocs.io/en/latest/evaluation.html)** | Turnkey evaluation on standard collections: MS MARCO, BEIR, NanoBEIR, LoTTE, Robust04, TREC-DL via `ir-measures`. |
| **[Datasets & Collections](https://experimaestro-ir.readthedocs.io/en/latest/data/index.html)** | Datamaestro and `ir-datasets` adapters, document tokenization, collection partitioning. |
| **[Experiments & Distributed](https://experimaestro-ir.readthedocs.io/en/latest/experiments.html)** | Declarative pipelines, multi-GPU Lightning Fabric training, SLURM cluster scheduling. |

---

## 🛠️ CLI & Experiment Management

`xpmir` seamlessly integrates with Experimaestro's command line to run declarative YAML or Python experiments:

```bash
# Validate task dependencies and launchers before running (Dry Run)
uv run experimaestro run configs/experiment.yaml --run-mode DRY_RUN

# Launch locally or submit to a SLURM cluster
uv run experimaestro run configs/experiment.yaml
```

To inspect running tasks, monitors, and outputs in real time:
```bash
experimaestro monitor
```

---

## 📖 Documentation & Resources

- **Full Documentation**: [https://experimaestro-ir.readthedocs.io/](https://experimaestro-ir.readthedocs.io/)
- **Projects & Research**: [xpmir GitHub Organization](https://github.com/xpmir) (featured research repositories and paper reproductions built with `xpmir`)
- **Core Orchestrator**: [Experimaestro](https://github.com/experimaestro/experimaestro-python)
- **PyTorch Foundations**: [xpm-torch](https://github.com/experimaestro/xpm-torch)
- **Dataset Ecosystem**: [datamaestro](https://github.com/experimaestro/datamaestro) & [datamaestro-ir](https://github.com/experimaestro/datamaestro-ir)
- **Reference Implementations**: [Reproducing Cross-Encoders](https://github.com/xpmir/cross-encoders)

---

## 📑 Citation

If you use `xpmir` in your research, please cite:

```bibtex
@software{piwowarski2023xpmir,
  author = {Benjamin Piwowarski and contributors},
  title = {Experimaestro IR: A Modular Framework for Information Retrieval Research},
  url = {https://github.com/experimaestro/experimaestro-ir},
  year = {2023}
}
```

## 📄 License

This project is licensed under the GNU General Public License v3.0 ([GPL-3.0](LICENSE)).
