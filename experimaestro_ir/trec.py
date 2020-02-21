from pathlib import Path
from experimaestro import config, task, pathargument, argument
import experimaestro_ir as ir

NS = ir.NS.trec


@argument("base", type=TrecSearchResults)
@task(parents=TrecSearchResults)
def Reorder(results: TrecSearchResults, base: TrecSearchResults):
    pass
