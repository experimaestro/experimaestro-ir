import inspect
import logging
import sys
from itertools import chain
from attrs import define
from datamaestro_text.interfaces.trec import write_run_dict
import pandas as pd
from pathlib import Path
from typing import DefaultDict, Dict, List, Protocol, Union, Tuple, Optional
import ir_measures
from experimaestro import Task, Param, pathgenerator, Annotated, tags, TagDict
from datamaestro_text.data.ir import (
    Adhoc,
    AdhocAssessments,
    AdhocRun,
    AdhocRunDict,
    Documents,
    AdhocResults,
    IDItem,
)
from datamaestro_text.data.ir.trec import TrecAdhocRun, TrecAdhocResults
from datamaestro_text.transforms.ir import TopicWrapper
from xpmir.measures import Measure
import xpmir.measures as m
from xpmir.metrics import evaluator
from xpmir.rankers import Retriever
from xpmir.utils.logging import easylog
from experimaestro.launchers import Launcher

logger = easylog()


def get_evaluator(metrics: List[ir_measures.Metric], assessments: AdhocAssessments):
    qrels = {
        assessedTopic.topic_id: {r.doc_id: r.rel for r in assessedTopic.assessments}
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

    with_run: Param[bool] = False
    """Saves the run together with the evaluation"""

    run_path: Annotated[Path, pathgenerator("run.txt")]
    """Path to save the run (TREC format). Only used if with_run is True"""

    def task_outputs(self, dep):
        results = dep(
            TrecAdhocResults.C(
                id="",
                results=self.aggregated,
                detailed=self.detailed,
                metrics=self.measures,
            )
        )
        if self.with_run:
            return results, dep(TrecAdhocRun.C(id="", path=self.run_path))
        return results

    def _execute(self, run: AdhocRunDict, assessments):
        """Evaluate an IR ad-hoc run with trec-eval"""

        if self.with_run:
            logging.info("Writing the run")
            write_run_dict(run, self.run_path)

        evaluator = get_evaluator([measure() for measure in self.measures], assessments)

        def print_line(fp, measure, scope, value):
            fp.write("{:25s} {:10s} {:.4f}\n".format(measure, scope, value))

        with self.detailed.open("w") as fp:
            for metric in evaluator.iter_calc(run):
                print_line(fp, str(metric.measure), metric.query_id, metric.value)

        with self.aggregated.open("w") as fp:
            for key, value in evaluator.calc_aggregate(run).items():
                print_line(fp, str(key), "all", value)


def get_run(retriever: Retriever, dataset: Adhoc) -> AdhocRunDict:
    """Returns the scored documents for each topic in a dataset"""
    results = retriever.retrieve_all(
        {topic[IDItem].id: topic for topic in dataset.topics.iter()}
    )
    return {
        qid: {sd.document[IDItem].id: sd.score for sd in scoredocs}
        for qid, scoredocs in results.items()
    }


def evaluate(retriever: Retriever, dataset: Adhoc, measures: List[str], details=False):
    """Evaluate a retriever on a given dataset

    :param retriever: The retriever to evaluate
    :param dataset: The dataset on which to evaluate
    :param measures: The list of measures to compute (using ir_measures)
    :param details: if query-level metrics should be reported, defaults to False
    :return: The metrics (if details is False) or a tuple (metrics, detailed metrics)
    """
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

    return metrics


class RunEvaluation(BaseEvaluation, Task):
    """Evaluate a run"""

    run: Param[TrecAdhocRun]
    assessments: Param[AdhocAssessments]

    def execute(self):
        run = ir_measures.read_trec_run(self.run.path, self.assessments)
        return self._execute(run)


class Evaluate(BaseEvaluation, Task):
    """Evaluate a retriever directly (without generating the run explicitly)"""

    dataset: Param[Adhoc]
    """The dataset for retrieval"""

    retriever: Param[Retriever]
    """The retriever to evaluate"""

    topic_wrapper: Param[Optional[TopicWrapper]] = None
    """Topic extractor"""

    def execute(self):
        self.retriever.initialize()
        run = get_run(self.retriever, self.dataset)
        self._execute(run, self.dataset.assessments)


class RetrieverFactory(Protocol):
    """Generates a retriever for a given dataset"""

    def __call__(self, dataset: Documents) -> Retriever:
        ...


class Evaluations:
    """Holds experiment results for several models
    on one dataset"""

    dataset: Adhoc
    measures: List[Measure]
    results: List[BaseEvaluation]
    per_tag: Dict[TagDict, AdhocResults]

    topic_wrapper: Optional[TopicWrapper]

    def __init__(
        self,
        dataset: Adhoc,
        measures: List[Measure],
        *,
        topic_wrapper: Optional[TopicWrapper] = None,
    ):
        self.dataset = dataset
        self.measures = measures
        self.results = []
        self.per_tags = {}
        self.topic_wrapper = topic_wrapper

    def evaluate_retriever(
        self,
        key: str,
        retriever: Union[Retriever, RetrieverFactory],
        launcher: Launcher = None,
        *,
        init_tasks=[],
        with_run=False,
    ) -> "EvaluationResult":
        """Evaluates a retriever

        :param key: test collection key
        :param retriever: the retriever (or the retriever factory)
        """
        if not isinstance(retriever, Retriever):
            kwargs = {}
            sig = inspect.signature(retriever)
            kw_only = set(
                name
                for name, p in sig.parameters.items()
                if p.kind == inspect.Parameter.KEYWORD_ONLY
            )
            if "key" in kw_only:
                kwargs["key"] = key
            retriever = retriever(self.dataset.documents, **kwargs)

        task = Evaluate.C(
            retriever=retriever,
            measures=self.measures,
            dataset=self.dataset,
            topic_wrapper=self.topic_wrapper,
            with_run=with_run,
        )

        evaluation = task.submit(launcher=launcher, init_tasks=init_tasks)

        run = None
        if with_run:
            evaluation, run = evaluation

        self.add(evaluation)

        # Use retriever tags
        retriever_tags = tags(evaluation)
        if retriever_tags:
            self.per_tags[retriever_tags] = evaluation

        return EvaluationResult(key, evaluation, run, task)

    def add(self, *results: BaseEvaluation):
        self.results.extend(results)

    def output_results_per_tag(self, file=sys.stdout):
        return self.to_dataframe().to_markdown(file)

    def to_dataframe(self) -> pd.DataFrame:
        # Get all the tags
        tags = list(
            set(chain(*[tags_dict.keys() for tags_dict in self.per_tags.keys()]))
        )
        tags.sort()

        assert (
            len(tags) > 0
        ), "No tags found, please tag your evaluations to convert results to dataframe"

        # Get all the results and metrics
        to_process = []
        metrics = set()
        for tags_dict, evaluate in self.per_tags.items():
            try:
                results = evaluate.get_results()
                metrics.update(results.keys())
                to_process.append((tags_dict, results))
            except FileNotFoundError:
                logger.error("Cannot retrieve evaluation results for %s", tags_dict)

        # Sort metrics
        metrics = list(metrics)
        metrics.sort()

        # Table header
        columns = []
        for tag in tags:
            columns.append(["tag", tag])
        for metric in metrics:
            columns.append(["metric", metric])

        # Output the results
        rows = []
        for tags_dict, results in to_process:
            row = []
            # tag values
            for k in tags:
                row.append(str(tags_dict.get(k, "")))

            # metric values
            for metric in metrics:
                row.append(results.get(metric, ""))
            rows.append(row)

        index = pd.MultiIndex.from_tuples(columns)
        return pd.DataFrame(rows, columns=index)


@define
class EvaluationResult:
    key: str
    """Dataset identifier"""

    result: AdhocResults
    """Results"""

    run: Optional[AdhocRun]
    """The run (if available)"""

    task: Evaluate
    """The task for this result"""


class EvaluationsCollection:
    """A collection of evaluation

    This is useful to group all the evaluations to be conducted, and then
    to call the :py:meth:`evaluate_retriever`
    """

    collection: Dict[str, Evaluations]

    per_model: Dict[str, List[Tuple[str, AdhocResults]]]
    """List of results per model"""

    def __init__(self, **collection: Evaluations):
        self.collection = collection
        self.per_model = {}

    def evaluate_retriever(
        self,
        retriever: Union[Retriever, RetrieverFactory],
        launcher: Launcher = None,
        model_id: str = None,
        overwrite: bool = False,
        with_run: bool = False,
        init_tasks=[],
    ) -> list[EvaluationResult]:
        """Evaluate a retriever for all the evaluations in this collection (the
        tasks are submitted to the experimaestro scheduler)

        :param with_run: should the run be preserved (default False). Note that
            this changes the experiment ID.
        """
        if model_id is not None and not overwrite:
            assert (
                model_id not in self.per_model
            ), f"Model with ID `{model_id}` was already evaluated"

        results = []
        for key, evaluations in self.collection.items():
            result = evaluations.evaluate_retriever(
                key, retriever, launcher, init_tasks=init_tasks, with_run=with_run
            )
            results.append(result)

        # Adds to per model results
        if model_id is not None:
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

    def to_dataframe(self) -> pd.DataFrame:
        """Returns a Pandas dataframe"""
        all_data = []
        for key, evaluations in self.collection.items():
            data = evaluations.to_dataframe()
            data["dataset"] = key
            all_data.append(data)
        return pd.concat(all_data, ignore_index=True)

    def output_results_per_tag(self, file=sys.stdout):
        """Outputs the results for each collection, based on the retriever tags
        to build the table
        """
        # Loop over all collections
        for key, evaluations in self.collection.items():
            print(f"## Dataset {key}\n", file=file)  # noqa: T201
            evaluations.output_results_per_tag(file)

    def output_model_results(self, model_id: str, file=sys.stdout):
        """Outputs the result of a model over various datasets (in markdown format)

        :param model_id: The model id, as given by :meth:`evaluate_retriever`
        :param file: The output stream, defaults to sys.stdout
        """
        all_results = {}
        all_metrics = set()
        for key, evaluation in self.per_model[model_id]:
            all_results[key] = evaluation.get_results()
            all_metrics.update(all_results[key].keys())
        all_metrics = sorted(all_metrics)

        file.write(f"| Dataset  | {' | '.join(all_metrics)}  |\n")  # noqa: E221
        file.write(f"|----| {'---|---'.join('' for _ in all_metrics)}---|\n")
        for key, values in all_results.items():
            file.write(f"| {key}")
            for metric in all_metrics:
                value = values.get(metric, "")
                file.write(f" | {value}")
            file.write(" |\n")
