from experimaestro import argument, task, pathoption, config
import experimaestro_ir as ir
from datamaestro_text.data.ir.trec import TrecAdhocAssessments, TrecAdhocRun, TrecAdhocResults

import logging
import pytrec_eval



@argument("assessments", TrecAdhocAssessments)
@argument("run", TrecAdhocRun)
@pathoption("aggregated", "aggregated.dat")
@pathoption("detailed", "detailed.dat")
@task(ir.NS.evaluate.trec)
class TrecEval():
    def config(self):
        return TrecAdhocResults(aggregated=self.aggregated, detailed=self.detailed)

    def execute(self):
        """Evaluate an IR ad-hoc run with trec-eval"""
        logging.info("Reading assessments %s", self.assessments.path)
        with self.assessments.path.open("r") as f_qrel:
            qrel = pytrec_eval.parse_qrel(f_qrel)

        logging.info("Reading results %s", self.run.path)
        with self.run.path.open("r") as f_run:
            run = pytrec_eval.parse_run(f_run)

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
        results = evaluator.evaluate(run)

        def print_line(fp, measure, scope, value):
            fp.write("{:25s}{:8s}{:.4f}\n".format(measure, scope, value))

        with self.detailed.open("w") as fp:
            for query_id, query_measures in sorted(results.items()):
                for measure, value in sorted(query_measures.items()):
                    print_line(fp, measure, query_id, value)

        # Scope hack: use query_measures of last item in previous loop to
        # figure out all unique measure names.
        #
        # TODO(cvangysel): add member to RelevanceEvaluator
        #                  with a list of measure names.
        with self.aggregated.open("w") as fp:
            for measure in sorted(query_measures.keys()):
                print_line(
                    fp,
                    measure,
                    "all",
                    pytrec_eval.compute_aggregated_measure(
                        measure,
                        [query_measures[measure] for query_measures in results.values()],
                    ),
                )
