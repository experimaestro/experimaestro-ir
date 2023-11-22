[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Documentation Status](https://readthedocs.org/projects/experimaestro-ir/badge/?version=latest)](https://experimaestro-ir.readthedocs.io/en/latest/?badge=latest)

# Information Retrieval for experimaestro

Information Retrieval module for [experimaestro](https://experimaestro-python.readthedocs.io/)

The full documentation can be read at [IR@experimaestro](https://experimaestro-ir.readthedocs.io/).

Finally, you can find the [roadmap](https://github.com/experimaestro/experimaestro-ir/issues/9).

## Install

Base experimaestro-IR can be installed with `pip install xpmir`.
Functionalities can be added by installing optional dependencies:

- `pip install xpmir[neural]` to install neural-IR packages (torch, etc.)
- `pip install xpmir[anserini]` to install Anserini related packages

For the development version, you can:

- If you just want the development version: install with `pip install git+https://github.com/experimaestro/experimaestro-ir.git`
- If you want to edit the code: clone and then do a `pip install -e .` within the directory

## What's inside?

- Collection management (using datamaestro)
    - Interface for the [IR datasets library](https://ir-datasets.com/)
    - Splitting IR datasets
    - Shuffling training triplets
- Representation
    - Word Embeddings
    - HuggingFace transformers
- Indices
    - dense: [FAISS](https://github.com/facebookresearch/faiss) interface
    - sparse: [xpmir-rust library](https://github.com/experimaestro/experimaestro-ir-rust)
- Standard Indexing and Retrieval
    - Anserini
- Learning to Rank
    - Pointwise
    - Pairwise
    - Distillation
- Neural IR
    - Cross-Encoder
    - Splade
    - DRMM
    - ColBERT
- Paper reproduction:
    - *MonoBERT* (Passage Re-ranking with BERT. Rodrigo Nogueira and Kyunghyun Cho. 2019)
    - (alpha) *DuoBERT* (Multi-Stage Document Ranking with BERT. Rodrigo Nogueira, Wei Yang, Kyunghyun Cho, Jimmy Lin. 2019)
    - (beta) *Splade v2* (SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval, Thibault Formal, Carlos Lassance, Benjamin Piwowarski, and Stéphane Clinchant. SIGIR 2021)
    - (planned) ANCE
- Pre-trained models
    - [HuggingFace](https://huggingface.co) [integration](https://experimaestro-ir.readthedocs.io/en/latest/pretrained.html) (direct, through the Sentence Transformers library)

## Thanks

Some parts of the code have been adapted from [OpenNIR](https://github.com/Georgetown-IR-Lab/OpenNIR)
