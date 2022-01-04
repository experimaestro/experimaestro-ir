"""Complements ir-metrics package"""

from typing import Iterator, Iterable, List
import ir_measures
from ir_measures import measures, Metric, ScoredDoc
from ir_measures.providers.base import NOT_PROVIDED, Evaluator, Any
import numpy as np


class _Accuracy(measures.Measure):
    """Accuracy metric

    Reports the probability that a relevant document is ranked before a non relevant one
    """

    __name__ = "accuracy"
    NAME = __name__
    SUPPORTED_PARAMS = {
        "cutoff": measures.ParamInfo(
            dtype=int, required=False, desc="ranking cutoff threshold"
        ),
        "rel": measures.ParamInfo(
            dtype=int,
            default=1,
            desc="minimum relevance score to be considered relevant (inclusive)",
        ),
    }

    @staticmethod
    def compute(nonrels: List[int]) -> float:
        if len(nonrels) < 2:
            return 1
        return 1.0 - np.mean(nonrels[:-1]) / float(nonrels[-1])


measures.register(_Accuracy())


class XPMIREvaluator(Evaluator):
    def __init__(self, measures, qrels):
        self.qrels = ir_measures.util.QrelsConverter(qrels).as_dict_of_dict()
        super().__init__(measures, set(self.qrels.keys()))

    def _iter_calc(self, run) -> Iterator["Metric"]:
        iter = ir_measures.util.RunConverter(
            run
        ).as_sorted_namedtuple_iter()  # type: Iterable[ScoredDoc]

        for measure in self.measures:
            cutoff = 0 if measure["cutoff"] is NOT_PROVIDED else measure["cutoff"]
            assert measure.NAME == _Accuracy.NAME

            # Builds a run
            qid = None
            nonrels = []
            qrels = {}
            rank = 0
            for scoreddoc in iter:
                if qid is None or qid != scoreddoc.query_id:
                    if qid is not None:
                        yield Metric(
                            query_id=qid,
                            measure=measure,
                            value=_Accuracy.compute(nonrels),
                        )
                    qid = scoreddoc.query_id
                    nonrels = [0]
                    rank = 0
                    qrels = self.qrels.get(qid, {})

                if len(qrels) == 0:
                    continue

                rank += 1
                if rank <= cutoff or cutoff == 0:
                    if qrels.get(scoreddoc.doc_id, 0) >= measure["rel"]:
                        nonrels.append(nonrels[-1])
                    else:
                        nonrels[-1] += 1

            if qid is not None:
                yield Metric(
                    query_id=qid, measure=measure, value=_Accuracy.compute(nonrels)
                )


class XPMIRProvider(ir_measures.providers.Provider):
    """
    The base class for all measure providers (e.g., pytrec_eval, gdeval, etc.).
    """

    NAME = "xpmir"
    SUPPORTED_MEASURES = [
        _Accuracy(cutoff=Any(), rel=Any()),
    ]

    def __init__(self):
        super().__init__()
        self._is_available = True

    def _evaluator(self, measures, qrels) -> Evaluator:
        return XPMIREvaluator(measures, qrels)


# Extend the default pipeline from
DefaultPipeline = ir_measures.providers.FallbackProvider(
    ir_measures.DefaultPipeline.providers + [XPMIRProvider()]
)
evaluator = DefaultPipeline.evaluator
