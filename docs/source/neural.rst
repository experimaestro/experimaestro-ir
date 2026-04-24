Neural models
=============

XPMIR provides implementations of the main neural IR scoring architectures.
Each architecture computes relevance scores differently:

- **Dual models** encode queries and documents independently, enabling
  pre-computation of document representations for fast retrieval.
- **Dense models** (DotDense, CosineDense) are dual models that produce a
  single vector per input and score via dot product or cosine similarity.
- **Late-interaction models** (ColBERT) keep per-token representations and
  compute fine-grained token-level interactions at scoring time.
- **Sparse models** (SPLADE) produce sparse bag-of-words representations with
  learned term weights.
- **Cross-encoders** jointly encode the query-document pair for maximum
  accuracy, at the cost of not being able to pre-compute representations.

.. contents:: On this page
   :local:
   :depth: 2


Dual models
-----------

Dual models compute a separate representation for documents and queries, which
allows pre-computing document representations and scoring efficiently over large
collections.

.. autoxpmconfig:: xpmir.neural.DualRepresentationScorer
    :members: score_pairs, score_product
.. autoxpmconfig:: xpmir.neural.dual.DualVectorScorer
.. autoxpmconfig:: xpmir.neural.dual.DualModuleLoader


Training hooks
^^^^^^^^^^^^^^

Hooks that can be attached to dual models during training (e.g. for
regularisation).

.. autoxpmconfig:: xpmir.neural.dual.DualVectorListener
    :members: __call__

.. autoxpmconfig:: xpmir.neural.dual.DualVectorScorerListener

.. autoxpmconfig:: xpmir.neural.dual.FlopsRegularizer
.. autoxpmconfig:: xpmir.neural.dual.ScheduledFlopsRegularizer


Dense models
------------

Dense models produce a single fixed-size vector per query or document and score
with a dot product or cosine similarity. They are commonly initialised from
`Sentence Transformers <https://www.sbert.net/>`_ checkpoints.

.. autoxpmconfig:: xpmir.neural.dual.Dense
    :members: from_sentence_transformers

.. autoxpmconfig:: xpmir.neural.dual.DotDense
.. autoxpmconfig:: xpmir.neural.dual.CosineDense

Late-interaction (ColBERT)
--------------------------

ColBERT retains per-token embeddings and scores via late interaction (MaxSim).
This gives accuracy close to cross-encoders while still allowing document
representations to be pre-computed and indexed.

.. autoxpmconfig:: xpmir.neural.colbert.ColBERTEncoder
    :members: encode_queries, encode_documents, document_token_embeddings, query_token_embeddings, score_product, score_pairs

Sparse models (SPLADE)
----------------------

SPLADE-family models produce sparse representations with learned term weights.
Documents and queries are mapped to high-dimensional sparse vectors over the
vocabulary, enabling efficient inverted-index retrieval.

.. autoxpmconfig:: xpmir.neural.splade.SpladeTextEncoder
.. autoxpmconfig:: xpmir.neural.splade.SpladeScorer
.. autoxpmconfig:: xpmir.neural.splade.SpladeModuleLoader
.. autoxpmconfig:: xpmir.neural.splade.Aggregation
.. autoxpmconfig:: xpmir.neural.splade.MaxAggregation
.. autoxpmconfig:: xpmir.neural.splade.SumAggregation

Cross-encoders (HuggingFace)
-----------------------------

Cross-encoders jointly encode the query and document with a single transformer
pass, producing the most accurate relevance scores. They are typically used as
re-rankers in a multi-stage pipeline.

.. autoxpmconfig:: xpmir.neural.huggingface.HFCrossScorer
.. autoxpmconfig:: xpmir.neural.huggingface.HFQueryDocTokenizer
.. autoxpmconfig:: xpmir.neural.huggingface.CrossEncoderModuleLoader
.. autoxpmconfig:: xpmir.neural.huggingface.InitCEFromHFID
.. autoxpmconfig:: xpmir.text.huggingface.base.HFConfig
.. autoxpmconfig:: xpmir.text.huggingface.base.HFConfigID
.. autoxpmconfig:: xpmir.text.huggingface.base.HFModelInitBase
.. autoxpmconfig:: xpmir.text.huggingface.base.HFSequenceClassification

Cross-encoders (Sentence-Transformers)
-------------------------------------

XPMIR also supports cross-encoders via the `Sentence Transformers <https://sbert.net/docs/cross_encoder/usage/usage.html>`_
library. This is particularly useful for models that require specific chat templates
or prompt-based ranking (like some LLM-based rankers) that are natively supported
by Sentence-Transformers.

.. autoxpmconfig:: xpmir.neural.sentence_transformers.STCrossEncoder
.. autoxpmconfig:: xpmir.neural.sentence_transformers.InitSTCrossEncoder
