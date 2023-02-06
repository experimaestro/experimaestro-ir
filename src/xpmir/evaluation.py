from pathlib import Path
import sys
from typing import DefaultDict, Dict, List, Protocol, Union
from datamaestro_text.data.ir import Adhoc, AdhocAssessments, AdhocDocuments
from experimaestro import Task, Param, pathgenerator, Annotated
from datamaestro_text.data.ir.trec import (
    TrecAdhocRun,
    TrecAdhocResults,
)
from xpmir.measures import Measure
import xpmir.measures as m
from xpmir.metrics import evaluator
import ir_measures
from xpmir.rankers import Retriever

from experimaestro.launchers import Launcher


def get_evaluator(metrics: List[ir_measures.Metric], assessments: AdhocAssessments):
    qrels = {
        assessedTopic.qid: {r.docno: r.rel for r in assessedTopic.assessments}
        for assessedTopic in assessments.iter()
    }
    return evaluator(metrics, qrels)


class BaseEvaluation(Task):
    """Base class for evaluation tasks"""

    measures: Param[List[Measure]] = [m.AP, m.P @ 20, m.nDCG, m.nDCG @ 20, m.RR]
    """List of metrics"""

    aggregated: Annotated[Path, pathgenerator("aggregated.txt")]
    """Path for aggregated results"""

    detailed: Annotated[Path, pathgenerator("detailed.dat")]
    """Path for detailed results"""

    def config(self):
        return TrecAdhocResults(
            id="",
            results=self.aggregated,
            detailed=self.detailed,
            metrics=self.measures,
        )

    def _execute(self, run, assessments):
        """Evaluate an IR ad-hoc run with trec-eval"""

        evaluator = get_evaluator([m() for m in self.measures], assessments)

        def print_line(fp, measure, scope, value):
            fp.write("{:25s}{:8s}{:.4f}\n".format(measure, scope, value))

        with self.detailed.open("w") as fp:
            for metric in evaluator.iter_calc(run):
                print_line(fp, str(metric.measure), metric.query_id, metric.value)

        # TODO: work-around bug in pytrec_eval
        # https://github.com/terrierteam/ir_measures/issues/49
        evaluator = get_evaluator([m() for m in self.measures], assessments)

        with self.aggregated.open("w") as fp:
            for key, value in evaluator.calc_aggregate(run).items():
                print_line(fp, str(key), "all", value)


def get_run(retriever: Retriever, dataset: Adhoc):
    """Evaluate a retriever on a dataset"""
    results = retriever.retrieve_all(
        {topic.qid: topic.text for topic in dataset.topics.iter()}
    )
    return {
        qid: {sd.docid: sd.score for sd in scoredocs}
        for qid, scoredocs in results.items()
    }


def evaluate(retriever: Retriever, dataset: Adhoc, measures: List[str], details=False):
    evaluator = get_evaluator(
        [ir_measures.parse_measure(m) for m in measures], dataset.assessments
    )
    run = get_run(retriever, dataset)

    aggregators = {m: m.aggregator() for m in evaluator.measures}
    details = DefaultDict(lambda: {}) if details else None
    for metric in evaluator.iter_calc(run):
        aggregators[metric.measure].add(metric.value)
        if details is not None:
            details[str(metric.measure)][metric.query_id] = metric.value

    metrics = {str(m): agg.result() for m, agg in aggregators.items()}
    if details is not None:
        return metrics, details

    return details


class RunEvaluation(BaseEvaluation, Task):
    """Evaluate a run"""

    run: Param[TrecAdhocRun]
    assessments: Param[AdhocAssessments]

    def execute(self):
        run = ir_measures.read_trec_run(self.run.path, self.assessments)
        return self._execute(run)


class Evaluate(BaseEvaluation, Task):
    """Evaluate a retriever directly (without generating the run explicitely)"""

    dataset: Param[Adhoc]
    """The dataset for retrieval"""

    retriever: Param[Retriever]
    """The retriever to evaluate"""

    def execute(self):
        self.retriever.initialize()
        run = get_run(self.retriever, self.dataset)
        self._execute(run, self.dataset.assessments)


class RetrieverFactory(Protocol):
    """Generates a retriever for a given dataset"""

    def __call__(self, dataset: AdhocDocuments, **kwargs) -> Retriever:
        ...


class Evaluations:
    """Holds experiment results for several models
    on one dataset"""

    dataset: Adhoc
    measures: List[Measure]
    results: List[BaseEvaluation]

    def __init__(self, dataset: Adhoc, measures: List[Measure]):
        self.dataset = dataset
        self.measures = measures
        self.results = []

    def evaluate_retriever(
        self, retriever: Union[Retriever, RetrieverFactory], launcher: Launcher = None
    ):
        """Evaluates a retriever"""
        if not isinstance(retriever, Retriever):
            retriever = retriever(self.dataset.documents)

        evaluation = Evaluate(
            retriever=retriever, measures=self.measures, dataset=self.dataset
        ).submit(launcher=launcher)
        self.add(evaluation)
        return evaluation

    def add(self, *results: BaseEvaluation):
        self.results.extend(results)


class EvaluationsCollection:
    """A collection of evaluation

    This is useful to group all the evaluations to be conducted, and then
    to call the :py:meth:`evaluate_retriever`
    """

    collection: Dict[str, Evaluations]
    per_model: Dict[str, List[Evaluate]]

    def __init__(self, **collection: Evaluations):
        self.collection = collection
        self.per_model = {}

    def evaluate_retriever(
        self,
        retriever: Union[Retriever, RetrieverFactory],
        launcher: Launcher = None,
        model_id: str = None,
    ):
        """Evaluate a retriever for all the evaluations in this collection (the
        tasks are submitted to experimaestro the scheduler)"""
        results = []
        for key, evaluations in self.collection.items():
            result = evaluations.evaluate_retriever(retriever, launcher)
            results.append((key, evaluations, result))

        if model_id is not None:
            assert (
                model_id not in self.per_model
            ), f"Model with ID `{model_id}` was already evaluated"
            self.per_model[model_id] = results

        return results

    def output_results(self, file=sys.stdout):
        """Print all the results"""
        for key, dsevaluations in self.collection.items():
            print(f"## Dataset {key}\n", file=file)  # noqa: T201
            for evaluation in dsevaluations.results:
                with evaluation.results.open("rt") as fp:
                    results = [f"- {line}" for line in fp.readlines()]
                    results = "".join(results)

                print(  # noqa: T201
                    f"### Results for {evaluation.__xpm__.tags()}" f"""\n{results}\n""",
                    file=file,
                )

    def output_model_results(self, model_id: str, file=sys.stdout):
        """Outputs the result of a model over various datasets (in markdown format)

        :param model_id: The model id, as given by :meth:`evaluate_retriever`
        :param file: The output stream, defaults to sys.stdout
        """
        all_results = {}
        all_metrics = set()
        for key, _, evaluation in self.per_model[model_id]:
            all_results[key] = evaluation.get_results()
            all_metrics.update(all_results[key].keys())
        all_metrics = sorted(all_metrics)

        file.write(f"| Dataset  | {' | '.join(all_metrics)}  |\n")
        file.write(f"|----| {'---|---'.join('' for _ in all_metrics)}---|\n")
        for key, values in all_results.items():
            file.write(f"| {key}")
            for metric in all_metrics:
                value = values.get(metric, "")
                file.write(f" | {value}")
            file.write(" |\n")
