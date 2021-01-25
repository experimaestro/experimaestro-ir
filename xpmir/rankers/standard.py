from experimaestro import Param, config


@config()
class Model:
    """Base class for standard IR models"""

    pass


@config()
class BM25(Model):
    k1: Param[float] = 0.9
    b: Param[float] = 0.4
