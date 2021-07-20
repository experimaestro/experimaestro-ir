from pathlib import Path
import tempfile
from typing import Iterator, List, Optional
from datamaestro_text.data.ir import Adhoc, AdhocAssessments
from experimaestro import Config, tqdm, Task, Param, pathgenerator, Annotated
from datamaestro_text.data.ir.trec import (
    TrecAdhocAssessments,
    TrecAdhocRun,
    TrecAdhocResults,
)

import ir_measures
from xpmir.rankers import Retriever


def get_evaluator(assessments: AdhocAssessments, measures: List[str]):
    class Qrels:
        """Adapter for ir_measures.Qrels"""

        def __iter__(self) -> Iterator[ir_measures.util.Qrel]:
            for qdata in assessments.iter():
                for assessment in qdata.assessments:
                    yield ir_measures.util.Qrel(
                        qdata.qid, assessment.docno, int(assessment.rel)
                    )

    return ir_measures.evaluator(
        [ir_measures.parse_measure(m) for m in measures], Qrels()
    )


class BaseEvaluation(Config):
    metrics: Param[List[str]] = ["MAP", "P@20", "NDCG", "", "mrr"]
    detailed: Annotated[Path, pathgenerator("detailed.txt")]
    measures: Annotated[Path, pathgenerator("measures.txt")]

    def print_results(self, evaluator: ir_measures.providers.Evaluator, run):
        def print_line(fp, measure, scope, value):
            fp.write("{:25s}{:8s}{:.4f}\n".format(measure, scope, value))

        with open(self.measures, "wt") as fp:
            mean_metrics = evaluator.calc_aggregate(run)
            for metric, value in mean_metrics.items():
                print_line(fp, str(metric), "all", value)

        with open(self.detailed, "wt") as fp:
            for metric in evaluator.iter_calc(run):
                print_line(fp, str(metric.measure), metric.query_id, metric.value)


class TrecEval(Task, BaseEvaluation):
    """Evaluate a retrieved of documents"""

    assessments: Param[TrecAdhocAssessments]
    run: Param[TrecAdhocRun]

    def config(self):
        return TrecAdhocResults(
            results=self.aggregated, detailed=self.detailed, metrics=self.metrics
        )

    def execute(self):
        """Evaluate an IR ad-hoc run with trec-eval"""

        evaluator = get_evaluator(self.assessments, self.metrics)
        run = ir_measures.read_trec_run(self.assessments.path)
        self.print_results(evaluator, run)


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
    evaluator = get_evaluator(dataset.assessments, measures)
    run = get_run(retriever, dataset)
    return {str(key): value for key, value in evaluator.calc_aggregate(run)}


class Evaluate(BaseEvaluation, Task):
    """Evaluate a retriever

    Attributes:

        metrics: the list of metrics
    """

    dataset: Param[Adhoc]
    retriever: Param[Retriever]

    def config(self):
        return TrecAdhocResults(
            results=self.measures, detailed=self.detailed, metrics=self.metrics
        )

    def execute(self):
        # Run the model
        evaluator = get_evaluator(self.dataset.assessments, self.metrics)

        self.retriever.initialize()
        run = get_run(self.retriever, self.dataset)

        self.print_results(evaluator, run)
