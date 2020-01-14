from pathlib import Path
from experimaestro import config, task, pathargument, argument
from datamaestro_text import NAMESPACE

NS = NAMESPACE.trec

@pathargument("results", "results.trec")
@config(NS.searchresults)
class TrecSearchResults:
    """Search results in the TREC format"""
    pass

@argument("base", type=TrecSearchResults)
@task(parents=TrecSearchResults)
def Reorder(results: TrecSearchResults, base: TrecSearchResults):
    pass