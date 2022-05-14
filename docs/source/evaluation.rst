Evaluation
==========



Evaluation
----------

.. autoxpmconfig:: xpmir.evaluation.Evaluate


Metrics
-------

Metrics are backed up by the module ir_measures

.. autoxpmconfig:: xpmir.measures.Measure

List of defined measures

.. automodule:: xpmir.measures
    :members: AP, P, RR, nDCG

Measures can be used with the @ operator. Exemple:

.. code-block:: python

    from xpmir.measures import AP, P, nDCG, RR
    from xpmir.evaluation import Evaluate

    Evaluate(measures=[AP, P@20, nDCG, nDCG@10, nDCG@20, RR, RR@10], ...)