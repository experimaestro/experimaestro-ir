
Neural models
-------------

Cross-Encoder
=============

Models that rely on a joint representation of the query and the document.

.. autoxpmconfig:: xpmir.neural.cross.CrossScorer
.. autoxpmconfig:: xpmir.neural.jointclassifier.JointClassifier

.. autoxpmconfig:: xpmir.neural.cross.DuoCrossScorer


Dual models
===========

Dual models compute a separate representation for documents
and queries, which allows some speedup when computing scores
of several documents and/or queries.


.. autoxpmconfig:: xpmir.neural.DualRepresentationScorer
    :members: score_pairs, score_product
.. autoxpmconfig:: xpmir.neural.dual.DualVectorScorer


Hooks
*****

.. autoxpmconfig:: xpmir.neural.dual.DualVectorListener
    :members: __call__


.. autoxpmconfig:: xpmir.neural.dual.FlopsRegularizer
.. autoxpmconfig:: xpmir.neural.dual.ScheduledFlopsRegularizer


Dense models
============


.. autoxpmconfig:: xpmir.neural.dual.Dense
    :members: from_sentence_transformers

.. autoxpmconfig:: xpmir.neural.dual.DotDense
.. autoxpmconfig:: xpmir.neural.dual.CosineDense

Interaction models
==================

.. autosummary::
    :nosignatures:

    xpmir.neural.interaction.InteractionScorer
    xpmir.neural.interaction.drmm.Drmm
    xpmir.neural.interaction.colbert.Colbert

.. autoxpmconfig:: xpmir.neural.interaction.InteractionScorer
.. autoxpmconfig:: xpmir.neural.interaction.drmm.Drmm

.. autoxpmconfig:: xpmir.neural.interaction.colbert.Colbert

DRMM
****

.. autoxpmconfig:: xpmir.neural.interaction.drmm.Combination
.. autoxpmconfig:: xpmir.neural.interaction.drmm.CountHistogram
.. autoxpmconfig:: xpmir.neural.interaction.drmm.IdfCombination
.. autoxpmconfig:: xpmir.neural.interaction.drmm.LogCountHistogram
.. autoxpmconfig:: xpmir.neural.interaction.drmm.NormalizedHistogram
.. autoxpmconfig:: xpmir.neural.interaction.drmm.SumCombination

Similarity
==========

.. autoxpmconfig:: xpmir.neural.interaction.common.Similarity
.. autoxpmconfig:: xpmir.neural.interaction.common.DotProductSimilarity
.. autoxpmconfig:: xpmir.neural.interaction.common.CosineSimilarity

.. autoclass:: xpmir.neural.interaction.common.SimilarityInput
.. autoclass:: xpmir.neural.interaction.common.SimilarityOutput


Sparse Models
=============

.. autoxpmconfig:: xpmir.neural.splade.SpladeTextEncoder
.. autoxpmconfig:: xpmir.neural.splade.SpladeTextEncoderV2
.. autoxpmconfig:: xpmir.neural.splade.Aggregation
.. autoxpmconfig:: xpmir.neural.splade.MaxAggregation
.. autoxpmconfig:: xpmir.neural.splade.SumAggregation

Generative Models
=================


.. autoxpmconfig:: xpmir.neural.generative.ConditionalGenerator

.. autoxpmconfig:: xpmir.neural.generative.cross.GenerativeCrossScorer

HuggingFace Generative Models
*****************************

.. autoxpmconfig:: xpmir.neural.generative.hf.LoadFromT5
.. autoxpmconfig:: xpmir.neural.generative.hf.T5IdentifierGenerator
.. autoxpmconfig:: xpmir.neural.generative.hf.T5ConditionalGenerator
.. autoxpmconfig:: xpmir.neural.generative.hf.T5CustomOutputGenerator

From Huggingface
================

.. autoxpmconfig:: xpmir.neural.huggingface.HFCrossScorer
