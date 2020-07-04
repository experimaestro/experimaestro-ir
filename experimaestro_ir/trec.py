from typing import List
from pathlib import Path
from experimaestro import config, task, option, pathoption, argument
from datamaestro_text.data.ir.trec import TrecAdhocRun
import experimaestro_ir as ir

NS = ir.NS.trec


@argument("base", type=TrecAdhocRun)
@task(parents=TrecAdhocRun)
def Reorder(results: TrecAdhocRun, base: TrecAdhocRun):
    pass
