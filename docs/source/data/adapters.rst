Dataset adapters
================

Adapters derive new datasets from existing ones by subsampling documents,
topics, or relevance assessments. They are useful for creating train/validation
splits, restricting evaluation to a subset of topics, or building
retriever-based collections for re-ranking experiments.

.. currentmodule:: xpmir.datasets.adapters

.. contents:: On this page
   :local:
   :depth: 2

Adhoc datasets
--------------

Split or combine ad-hoc retrieval datasets into folds.

.. autoxpmconfig:: RandomFold
    :members: folds

.. autoxpmconfig:: ConcatFold

Documents
---------

Create document subsets, e.g. restricting a collection to documents returned by
a first-stage retriever.

.. autoxpmconfig:: RetrieverBasedCollection

.. autoxpmconfig:: DocumentSubset

Assessments
-----------

Fold relevance assessments (qrels) by topic ID or topic object.

.. autoxpmconfig:: AbstractAdhocAssessmentFold
.. autoxpmconfig:: AdhocAssessmentFold
.. autoxpmconfig:: IDAdhocAssessmentFold

Topics
------

Fold or generate topic sets.

.. autoxpmconfig:: AbstractTopicFold
.. autoxpmconfig:: TopicFold
.. autoxpmconfig:: IDTopicFold
.. autoxpmconfig:: TopicsFoldGenerator
.. autoxpmconfig:: MemoryTopicStore
.. autoxpmconfig:: TextStore
