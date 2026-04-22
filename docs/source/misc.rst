Miscellaneous
=============

Additional utility classes that support the main IR pipeline.

.. contents:: On this page
   :local:
   :depth: 2


Data conversion
---------------

Converters transform between different data representations (e.g. converting
retriever output formats).

.. autoxpmconfig:: xpmir.utils.convert.Converter

ID lists
--------

Configurations that represent ordered lists of document or topic IDs, used for
filtering or subsetting collections.

.. autoxpmconfig:: xpmir.misc.IDList
.. autoxpmconfig:: xpmir.misc.FileIDList

Model export
------------

Actions for exporting trained models (e.g. to HuggingFace Hub).

.. autoxpmconfig:: xpmir.models.XPMIRExportAction


Validation
----------

Listeners that monitor model performance during training and control
early-stopping or best-model checkpointing.

.. autoxpmconfig:: xpmir.letor.validation.ValidationListener
.. autoxpmconfig:: xpmir.letor.validation.AggregatorValidationListener
.. autoxpmconfig:: xpmir.letor.validation.ValidationSettings


Processors
----------

Pre- and post-processing transforms applied to documents, queries, or records
before scoring.

.. autoxpmconfig:: xpmir.letor.processors.DocumentsProcessor
.. autoxpmconfig:: xpmir.letor.processors.QueriesProcessor
.. autoxpmconfig:: xpmir.letor.processors.RecordsProcessor


Listwise distillation
---------------------

Listwise distillation losses and trainers (see also :doc:`letor/distillation`
for pairwise distillation).

.. autoxpmconfig:: xpmir.letor.distillation.listwise.DistillationListwiseLoss
.. autoxpmconfig:: xpmir.letor.distillation.listwise.DistillationListwiseTrainer
.. autoxpmconfig:: xpmir.letor.distillation.listwise.ListwiseSoftmaxCrossEntropy
.. autoxpmconfig:: xpmir.letor.distillation.listwise.DistillRankNetLoss
.. autoxpmconfig:: xpmir.letor.distillation.listwise.ADR_MSE
.. autoxpmconfig:: xpmir.letor.distillation.samplers.DistillationListwiseSampler
.. autoxpmconfig:: xpmir.letor.distillation.samplers.DistillationNegativesSampler


Index utilities
---------------

Bag-of-words retrieval and sparse-to-BMP format conversion.

.. autoxpmconfig:: xpmir.index.bow.BOWRetriever
.. autoxpmconfig:: xpmir.index.bow.BOWSparseRetrieverIndex
.. autoxpmconfig:: xpmir.index.bow.BOWSparseRetrieverIndexBuilder
.. autoxpmconfig:: xpmir.index.sparse.Sparse2BMPConverter
