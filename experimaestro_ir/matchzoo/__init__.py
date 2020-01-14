from experimaestro import config, task, argument
from experimaestro_ir import NAMESPACE as NS

@config(NS.matchzoo.models.matchpyramid)
class MatchPyramid: pass

@argument("model")
@task(parents=NS.matchzoo.learnedmodel)
def MatchZooLearn():
    pass