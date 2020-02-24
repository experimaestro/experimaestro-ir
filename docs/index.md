# Welcome experimaestro-ir

## Install

Install with

```
pip install experimaestro_ir
```

Specific extra dependencies can be used if you plan to use some
specific part of this module, e.g. for neural models

```
pip install experimaestro_ir[neural]
```

## Example

Below is an example of a simple experiment that runs BM25 and then learn and test a DRMM model.

```py
--8<-- "examples/neural-ir.py"
```
