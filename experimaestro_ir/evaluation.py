from typing import List
from experimaestro import param, task, pathoption, config
import experimaestro_ir as ir
from datamaestro_text.data.ir.trec import (
    TrecAdhocAssessments,
    TrecAdhocRun,
    TrecAdhocResults,
)

import logging
import experimaestro_ir.metrics as metrics


@param("assessments", TrecAdhocAssessments)
@param("run", TrecAdhocRun)
@param("metrics", type=List[str], default=["map", "p@20", "ndcg", "ndcg@20", "mrr"])
@pathoption("aggregated", "aggregated.dat")
@pathoption("detailed", "detailed.dat")
@task(ir.NS.evaluate.trec)
class TrecEval:
    def config(self):
        return TrecAdhocResults(
            results=self.aggregated, detailed=self.detailed, metrics=self.metrics
        )

    def execute(self):
        """Evaluate an IR ad-hoc run with trec-eval"""

        detailed =  metrics.calc(str(self.assessments.path), str(self.run.path), self.metrics)
        means = metrics.mean(detailed)
        print(means)

        def print_line(fp, measure, scope, value):
            fp.write("{:25s}{:8s}{:.4f}\n".format(measure, scope, value))

        with self.detailed.open("w") as fp:
            for measure, values in detailed.items():
                for query_id, value in sorted(values.items()):
                        print_line(fp, measure, query_id, value)

        # Scope hack: use query_measures of last item in previous loop to
        # figure out all unique measure names.
        #
        # TODO(cvangysel): add member to RelevanceEvaluator
        #                  with a list of measure names.
        with self.aggregated.open("w") as fp:
            for measure, value in sorted(means.items()):
                print_line(
                    fp,
                    measure,
                    "all",
                    value
                )
