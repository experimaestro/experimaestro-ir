
Neural models
-------------

Dual models
===========

Dual models compute a separate representation for documents
and queries, which allows some speedup when computing scores
of several documents and/or queries.


.. autoxpmconfig:: xpmir.neural.DualRepresentationScorer
    :members: score_pairs, score_product
.. autoxpmconfig:: xpmir.neural.dual.DualVectorScorer
.. autoxpmconfig:: xpmir.neural.dual.DualModuleLoader


Hooks
*****

.. autoxpmconfig:: xpmir.neural.dual.DualVectorListener
    :members: __call__

.. autoxpmconfig:: xpmir.neural.dual.DualVectorScorerListener


.. autoxpmconfig:: xpmir.neural.dual.FlopsRegularizer
.. autoxpmconfig:: xpmir.neural.dual.ScheduledFlopsRegularizer


Dense models
============


.. autoxpmconfig:: xpmir.neural.dual.Dense
    :members: from_sentence_transformers

.. autoxpmconfig:: xpmir.neural.dual.DotDense
.. autoxpmconfig:: xpmir.neural.dual.CosineDense

Sparse Models
=============

.. autoxpmconfig:: xpmir.neural.splade.SpladeTextEncoder
.. autoxpmconfig:: xpmir.neural.splade.SpladeScorer
.. autoxpmconfig:: xpmir.neural.splade.SpladeModuleLoader
.. autoxpmconfig:: xpmir.neural.splade.Aggregation
.. autoxpmconfig:: xpmir.neural.splade.MaxAggregation
.. autoxpmconfig:: xpmir.neural.splade.SumAggregation

From Huggingface
================

.. autoxpmconfig:: xpmir.neural.huggingface.HFCrossScorer
.. autoxpmconfig:: xpmir.neural.huggingface.HFQueryDocTokenizer
.. autoxpmconfig:: xpmir.neural.huggingface.CrossEncoderModuleLoader
.. autoxpmconfig:: xpmir.neural.huggingface.InitCEFromHFID
.. autoxpmconfig:: xpmir.text.huggingface.base.HFConfig
.. autoxpmconfig:: xpmir.text.huggingface.base.HFConfigID
.. autoxpmconfig:: xpmir.text.huggingface.base.HFModelInitBase
.. autoxpmconfig:: xpmir.text.huggingface.base.HFSequenceClassification
