from experimaestro import argument, config


@config()
class Model:
    """Base class for standard IR models"""

    pass


@argument("k1", default=0.9)
@argument("b", default=0.4)
@config()
class BM25(Model):
    pass
