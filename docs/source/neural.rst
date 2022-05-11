
Neural models
-------------

Cross-Encoder
=============

Models that rely on a joint representation of the query and the document.

.. autoxpmconfig:: xpmir.neural.cross.CrossScorer


Dense models
============


.. autoxpmconfig:: xpmir.neural.DualRepresentationScorer
    :members: score_pairs, score_product
.. autoxpmconfig:: xpmir.neural.dual.Dense
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

Sparse Models
=============

.. autoxpmconfig:: xpmir.neural.splade.SpladeTextEncoder
.. autoxpmconfig:: xpmir.neural.splade.Aggregation
.. autoxpmconfig:: xpmir.neural.splade.MaxAggregation
.. autoxpmconfig:: xpmir.neural.splade.SumAggregation


Pretrained models
=================

.. automodule:: xpmir.neural.pretrained
    :members:
