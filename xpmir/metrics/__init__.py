"""Complements ir-metrics package"""

import builtins
from pkg_resources import parse_version
import ir_measures

if getattr(builtins, "__sphinx_build__", False):
    evaluator = None
else:
    evaluator = ir_measures.DefaultPipeline.evaluator

    # If new measures have to be defined, before they make it upstream:
    #
    # DefaultPipeline = ir_measures.providers.FallbackProvider(
    #     ir_measures.DefaultPipeline.providers + [AccuracyProvider()]
    # )
    # evaluator = DefaultPipeline.evaluator
