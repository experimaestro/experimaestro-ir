Batchwise learning
==================

In batchwise learning, the loss is computed over entire batches of documents
rather than individual pairs. This enables losses such as in-batch
contrastive learning or listwise softmax cross-entropy.

Trainer
-------

.. autoxpmconfig:: xpmir.letor.trainers.batchwise.BatchwiseTrainer

Samplers
--------

.. autoxpmconfig:: xpmir.documents.samplers.RandomSpanSampler
