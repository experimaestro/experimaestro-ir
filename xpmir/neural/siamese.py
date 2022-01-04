from experimaestro import deprecate
import xpmir.neural.dual as dual


@deprecate
class DotDense(dual.DotDense):
    pass


@deprecate
class CosineDense(dual.CosineDense):
    pass


@deprecate
class DenseDocumentEncoder(dual.DenseDocumentEncoder):
    pass
