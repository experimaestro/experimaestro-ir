from experimaestro import Param, Config


class Model(Config):
    """Base class for standard IR models"""

    pass


class BM25(Model):
    """BM-25 model definition"""

    k1: Param[float] = 0.9
    b: Param[float] = 0.4


class QLDirichlet(Model):
    """Query likelihood (Dirichlet smoothing) model definition"""

    mu: Param[float] = 1000
