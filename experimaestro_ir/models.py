from experimaestro import argument, config
import experimaestro_ir as ir

NS = ir.NS.models


@config(NS.model)
class Model:
    pass


@argument("k1", default=0.9)
@argument("b", default=0.4)
@config(NS.bm25)
class BM25(Model):
    pass
