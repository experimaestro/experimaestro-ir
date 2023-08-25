
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
    xpmir.neural.colbert.Colbert


.. autoxpmconfig:: xpmir.neural.interaction.InteractionScorer
.. autoxpmconfig:: xpmir.neural.interaction.drmm.Drmm

.. autoxpmconfig:: xpmir.neural.colbert.Colbert

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

.. autoxpmconfig:: xpmir.neural.common.Similarity
.. autoxpmconfig:: xpmir.neural.common.L2Distance
.. autoxpmconfig:: xpmir.neural.common.CosineSimilarity


Sparse Models
=============

.. autoxpmconfig:: xpmir.neural.splade.SpladeTextEncoder
.. autoxpmconfig:: xpmir.neural.splade.Aggregation
.. autoxpmconfig:: xpmir.neural.splade.MaxAggregation
.. autoxpmconfig:: xpmir.neural.splade.SumAggregation

From Huggingface
================

.. autoxpmconfig:: xpmir.neural.huggingface.HFCrossScorer
