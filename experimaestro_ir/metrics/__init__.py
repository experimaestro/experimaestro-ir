from typing import Dict

from experimaestro_ir.metrics.base import BaseMetrics, Metric
from experimaestro_ir.metrics.fallback import FallbackMetrics
from experimaestro_ir.metrics.gdeval import GdevalMetrics
from experimaestro_ir.metrics.judged import JudgedMetrics
from experimaestro_ir.metrics.msmarco import MsMarcoMetrics
from experimaestro_ir.metrics.pytreceval import PyTrecEvalMetrics
from experimaestro_ir.metrics.treceval import TrecEvalMetrics


primary = FallbackMetrics(
    [
        MsMarcoMetrics(),
        PyTrecEvalMetrics(),
        TrecEvalMetrics(),
        JudgedMetrics(),
        GdevalMetrics(),
    ]
)
calc = primary.calc_metrics


def mean(metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    result = {}
    for m_name, values_by_query in metrics.items():
        result[m_name] = sum(values_by_query.values()) / len(values_by_query)
    return result
