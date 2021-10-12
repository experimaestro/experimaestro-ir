.. Experimaestro documentation master file, created by
   sphinx-quickstart on Thu Oct  7 18:32:09 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Experimaestro IR documentation!
==========================================

.. toctree::
   :maxdepth: 2

   indexbased
   letor
   neural
   misc


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



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
