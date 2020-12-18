# Welcome experimaestro-ir

## Install

Install with

```
pip install xpmir
```

## Example

Below is an example of a simple experiment that runs BM25 and evaluates the run (on TREC-1).
Note that you need the dataset to be prepared using
```
datamaestro datafolders set gov.nist.trec.tipster TIPSTER_PATH
datamaestro prepare gov.nist.trec.adhoc.1
```
with `TIPSTER_PATH` the path containg the TIPSTER collection (i.e. the folders `Disk1`, `Disk2`, etc.)

Run

```py
--8<-- "examples/bm25.py"
```
