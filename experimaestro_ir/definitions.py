from pathlib import Path
from experimaestro import Identifier, config, pathargument

NAMESPACE = Identifier("ir")

@pathargument("results", "results.trec")
@config(NAMESPACE.searchresults.trec)
class TrecSearchResults: pass