from experimaestro import argument, task
import experimaestro_ir as ir
from datamaestro_text.data.trec import TrecAssessments

import pytrec_eval

@argument("assessments", TrecAssessments)
@argument("results", ir.TrecSearchResults)
@task(ir.NAMESPACE.evaluate.trec)
def TrecEval(): 
    """Evaluate an IR ad-hoc run with trec-eval"""
    pass