Welcome to Experimaestro IR documentation!
==========================================

experimaestro-IR (XPMIR) is a library for building and evaluating Information
Retrieval models, with a focus on neural approaches. XPMIR defines a large set
of composable components -- scorers, retrievers, text encoders, samplers, and
evaluation pipelines -- that can be combined to build reproducible experiments.

XPMIR is built upon `experimaestro <https://experimaestro-python.readthedocs.io/en/latest/>`_,
a framework that tracks parameters, manages dependencies, and executes
experimental plans locally or on a cluster.


Install
=======

Base experimaestro-IR can be installed with ``pip install xpmir``.
Optional dependencies unlock additional functionality:

- ``pip install xpmir[neural]`` -- PyTorch, Transformers, and Sentence
  Transformers for neural IR models
- ``pip install xpmir[anserini]`` -- Anserini/Pyserini for classical IR models


Example
=======

Below is a minimal experiment that indexes a collection, runs BM25, and
evaluates the results on TREC-1. First, prepare the dataset:

.. code-block:: sh

   datamaestro datafolders set gov.nist.trec.tipster TIPSTER_PATH
   datamaestro prepare gov.nist.trec.adhoc.1

where ``TIPSTER_PATH`` is the path containing the TIPSTER collection (i.e. the
folders ``Disk1``, ``Disk2``, etc.).

Then execute the following file:

.. literalinclude:: ../../examples/bm25.py


Table of Contents
=================

.. toctree::
   :maxdepth: 2

   data/index
   retrieval
   evaluation
   letor/index
   neural
   text/index
   hooks
   misc
   experiments
   pretrained
   cli


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
