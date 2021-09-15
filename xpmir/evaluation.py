from pathlib import Path
import tempfile
from typing import Iterator, List, Optional
from datamaestro_text.data.ir import Adhoc, AdhocAssessments
from experimaestro import Config, tqdm, Task, Param, pathgenerator, Annotated
from datamaestro_text.data.ir.trec import (
    TrecAdhocRun,
    TrecAdhocResults,
)

import ir_measures
from xpmir.rankers import Retriever


def get_evaluator(measures: List[str], assessments: AdhocAssessments):
    qrels = {
        assessedTopic.qid: {r.docno: r.rel for r in assessedTopic.assessments}
        for assessedTopic in assessments.iter()
    }
    metrics = [ir_measures.parse_measure(measures) for measures in measures]
    return ir_measures.evaluator(metrics, qrels)


class BaseEvaluation(Task):
    measures: Param[List[str]] = ["AP", "P@20", "NDCG", "NDCG@20", "RR"]
    aggregated: Annotated[Path, pathgenerator("aggregated.txt")]
    detailed: Annotated[Path, pathgenerator("detailed.dat")]

    def config(self):
        return TrecAdhocResults(
            results=self.aggregated, detailed=self.detailed, metrics=self.measures
        )

    def _execute(self, run, assessments):
        """Evaluate an IR ad-hoc run with trec-eval"""

        evaluator = get_evaluator(self.measures, assessments)

        def print_line(fp, measure, scope, value):
            fp.write("{:25s}{:8s}{:.4f}\n".format(measure, scope, value))

        with self.detailed.open("w") as fp:
            for metric in evaluator.iter_calc(run):
                print_line(fp, str(metric.measure), metric.query_id, metric.value)

        # Scope hack: use query_measures of last item in previous loop to
        # figure out all unique measure names.
        #
        # TODO(cvangysel): add member to RelevanceEvaluator
        #                  with a list of measure names.
        with self.aggregated.open("w") as fp:
            for key, value in evaluator.calc_aggregate(run).items():
                print_line(fp, str(key), "all", value)


def get_run(retriever: Retriever, dataset: Adhoc):
    """Evaluate a retriever on a dataset"""
    topics = list(dataset.topics.iter())

    run = {}
    for query in tqdm(topics):
        scoreddocs = {}
        run[query.qid] = scoreddocs

        for rank, sd in enumerate(retriever.retrieve(query.text)):
            scoreddocs[sd.docid] = sd.score

    return run


def evaluate(retriever: Retriever, dataset: Adhoc, measures: List[str]):
    evaluator = get_evaluator(measures, dataset.assessments)
    run = get_run(retriever, dataset)
    return {str(key): value for key, value in evaluator.calc_aggregate(run).items()}


class RunEvaluation(BaseEvaluation, Task):
    run: Param[TrecAdhocRun]
    assessments: Param[AdhocAssessments]

    def execute(self):
        run = ir_measures.read_trec_run(self.run.path, assessments)
        return self._execute(run)


class Evaluate(BaseEvaluation, Task):
    """Evaluate a retriever

    Attributes:

        metrics: the list of metrics
    """

    dataset: Param[Adhoc]
    retriever: Param[Retriever]

    def execute(self):
        run = get_run(self.retriever, self.dataset)
        self._execute(run, self.dataset.assessments)
