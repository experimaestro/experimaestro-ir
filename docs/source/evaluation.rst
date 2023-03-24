Evaluation
==========



Evaluation
----------

.. autoxpmconfig:: xpmir.evaluation.Evaluate

.. autoclass:: xpmir.evaluation.Evaluations
    :members: evaluate_retriever, add

.. autoclass:: xpmir.evaluation.EvaluationsCollection
    :members: evaluate_retriever

Metrics
-------

Metrics are backed up by the module ir_measures

.. autoxpmconfig:: xpmir.measures.Measure

List of defined measures

.. automodule:: xpmir.measures
    :members: AP, P, RR, nDCG, R

Measures can be used with the @ operator. Exemple:

.. code-block:: python

    from xpmir.measures import AP, P, nDCG, RR
    from xpmir.evaluation import Evaluate

    Evaluate(measures=[AP, P@20, nDCG, nDCG@10, nDCG@20, RR, RR@10], ...)
