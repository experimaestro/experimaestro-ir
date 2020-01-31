from pathlib import Path
from experimaestro import config, task, pathargument, argument
import experimaetro_ir as ir

NS = ir.trec

@argument("base", type=TrecSearchResults)
@task(parents=TrecSearchResults)
def Reorder(results: TrecSearchResults, base: TrecSearchResults):
    pass