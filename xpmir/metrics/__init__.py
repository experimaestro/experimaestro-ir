from typing import Dict

from xpmir.metrics.base import BaseMetrics, Metric
from xpmir.metrics.fallback import FallbackMetrics
from xpmir.metrics.gdeval import GdevalMetrics
from xpmir.metrics.judged import JudgedMetrics
from xpmir.metrics.msmarco import MsMarcoMetrics
from xpmir.metrics.pytreceval import PyTrecEvalMetrics
from xpmir.metrics.treceval import TrecEvalMetrics


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
