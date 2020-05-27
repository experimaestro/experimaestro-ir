# Welcome experimaestro-ir

## Install

Install with

```
pip install experimaestro_ir
```

## Example

Below is an example of a simple experiment that runs BM25 and evaluates the run (on TREC-1).
Note that you need the dataset to be prepared using `datamaestro prepare gov.nist.trec.adhoc.1`
(and that the TREC-1 document be available somewhere on your hard drive).

```py
--8<-- "examples/bm25.py"
```
