Welcome to Experimaestro IR documentation!
==========================================

experimaestro-IR (XPMIR) is a library for learning IR (neural) models.
XPMIR defines a large set of components that can be composed arbitrarily,
allowing to re-use easily components to build your own experiments.
XPMIR is built upon `experimaestro <https://experimaestro-python.readthedocs.io/en/latest/>`_,
a library which allows to build complex experimental plans while tracking parameters
and to execute them locally or on a cluster.


Install
=======

Base experimaestro-IR can be installed with `pip install xpmir`.
Functionalities can be added by installing optional dependencies:

- `pip install xpmir[neural]` to install neural-IR packages
- `pip install xpmir[anserini]` to install Anserini related packages



Example
=======


Below is an example of a simple experiment that runs BM25 and evaluates the run (on TREC-1).
Note that you need the dataset to be prepared using

.. code-block:: sh

   datamaestro datafolders set gov.nist.trec.tipster TIPSTER_PATH
   datamaestro prepare gov.nist.trec.adhoc.1

with `TIPSTER_PATH` the path containg the TIPSTER collection (i.e. the folders `Disk1`, `Disk2`, etc.)

You can then execute the following file:

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
   hooks
   text/index
   papers/index
   pretrained
   misc


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
