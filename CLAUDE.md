# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

experimaestro-IR (XPMIR) is a Python Information Retrieval framework built on [experimaestro](https://experimaestro-python.readthedocs.io/) and [xpm-torch](https://github.com/experimaestro/xpm-torch). It provides neural IR models (SPLADE, ColBERT, cross-encoders), learning-to-rank, indexing (FAISS, sparse, Anserini), and paper reproduction infrastructure. Published as `xpmir` on PyPI.

## Commands

This project uses [uv](https://docs.astral.sh/uv/) as the package manager. All Python commands should be run via `uv run` to use the project's virtual environment.

```sh
# Install dependencies (creates .venv automatically)
uv sync --group test

# Run all tests
uv run pytest

# Run specific test file or function
uv run pytest src/xpmir/test/neural/test_forward.py
uv run pytest src/xpmir/test/neural/test_forward.py::test_forward_types -vvs

# Lint and format (ruff is the primary tool)
uv run ruff check .
uv run ruff check --fix .
uv run ruff format .

# Initialize submodules after cloning
git submodule update --init --recursive
```

## Conventions

- **Commits**: Conventional commits enforced by pre-commit hook (e.g., `feat:`, `fix:`, `refactor:`, `chore:`)
- **Code style**: ruff for linting and formatting (max 88 chars). No `FIXME:` or `TODO:` markers in Python files (enforced by pre-commit)
- **Python**: Supports 3.10, 3.11, 3.12

## Architecture

### Core Pattern: Configuration-Driven Experiments

Classes use experimaestro's `@configuration()` decorator and inherit from `Config`. Parameters are declared as `Param[Type]` for YAML serialization and dependency injection. Configuration objects are instantiated via `.C()` (configuration) then `.instance()` (lazy initialization).

Experiments use `@ir_experiment()` or `@learning_experiment` decorators wrapping a single `run(xp, cfg)` function.

### Key Module Map (`src/xpmir/`)

| Module | Purpose |
|--------|---------|
| `rankers/` | Base `Scorer` and `Retriever` abstractions, two-stage retrievers |
| `neural/` | `DualRepresentationScorer`, `DotDense`, `CosineDense`, SPLADE, HuggingFace integration |
| `text/` | Text encoders, tokenizers, adapters (`TextEncoderBase`, `TokensEncoder`) |
| `index/` | FAISS dense indexing, sparse indexing (xpmir-rust), Anserini |
| `letor/` | Learning-to-rank: record types (Pointwise, Pairwise, Product), processors, validation, samplers |
| `datasets/` | Dataset adapters for ir-datasets |
| `evaluation.py` | Evaluation tasks using ir-measures |
| `papers/` | Paper reproduction helpers (MS MARCO, optimizer configs, samplers) |
| `experiments/` | `IRExperimentHelper`, `LearningExperimentHelper` |
| `context.py` | Hook system (`InitializationHook`, `Context`) |
| `models.py` | HuggingFace Hub integration (`AutoModel`) |
| `measures.py` | Metric definitions (AP, nDCG, RR, P@k) |

### Key Class Hierarchies

- **Scorers**: `Scorer` → `AbstractModuleScorer` → neural implementations; `DualRepresentationScorer[QueriesRep, DocsRep]` → `DualVectorScorer` → `DotDense`/`CosineDense`
- **Retrievers**: `Retriever` → `AnseriniRetriever`, `FaissRetriever`, `TwoStageRetriever`
- **Text Encoding**: `TextEncoderBase` → `TokensEncoder`, `TokenizedTextEncoderBase`, `DualTextEncoder`
- **Learning Records**: `ProductRecords` (all pairs), `PairwiseRecords` (ranking), `PointwiseRecords` (relevance)

### Training Integration

Built on `xpm_torch` / PyTorch Lightning. `TrainerContext` is passed through forward methods. `TrainingHook` system enables custom loss/regularization (e.g., `ScheduledFlopsRegularizer` for SPLADE).

### Submodule Dependencies

- `xpm-torch/` — PyTorch + experimaestro integration (local submodule)
- `datamaestro-text/` — Text dataset management (local directory, no longer a submodule)
- `ir_datasets` — Pinned to custom fork in some projects

### Test Patterns

- Parametrized model tests using `@registermodel` decorator and `modelfactories` list
- `skip_if_ci` marker from `xpmir.test` for expensive tests
- `conftest.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` for macOS compatibility

### Documentation

Config classes must be documented in the Sphinx RST files under `docs/source/` using the `autoxpmconfig` directive. The `test_documented` test verifies all `Config` subclasses are documented.

```rst
.. autoxpmconfig:: xpmir.module.path.ClassName
   :members: method1, method2
```

Use the **full module path** (matching `ClassName.__module__`), not re-export aliases. For example, use `xpmir.rankers.scorer.Scorer` not `xpmir.rankers.Scorer`.
