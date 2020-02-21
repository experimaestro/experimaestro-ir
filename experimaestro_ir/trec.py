from pathlib import Path
from experimaestro import config, task, pathargument, argument
from datamaestro_text.data.trec import TrecAdhocResults
import experimaestro_ir as ir

NS = ir.NS.trec


@argument("base", type=TrecAdhocResults)
@task(parents=TrecAdhocResults)
def Reorder(results: TrecAdhocResults, base: TrecAdhocResults):
    pass
