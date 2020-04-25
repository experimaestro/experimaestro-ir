from experimaestro import config, argument

from . import openNIR, Factory


@argument(
    "lossfn", type=str, default="softmax"
)  # ['softmax', 'cross_entropy', 'hinge'])
@argument("pos_source", type=str, default="intersect")  # ['intersect', 'qrels'])
@argument("neg_source", type=str, default="neg_source")  # ['run', 'qrels', 'union'])
@argument("sampling", type=str, default="query")  # ['query', 'qrel'])
@argument("pos_minrel", default=1)
@argument("unjudged_rel", default=0)
@argument("num_neg", default=1)
@argument("margin", default=0.0)
@config(openNIR.trainer)
class Trainer:
    pass


@argument("source", default="run")
@argument("lossfn", default="mse")
@argument("minrel", default=-999)
@config(openNIR.trainer.pointwise)
class PointwiseTrainer(Factory):
    pass
