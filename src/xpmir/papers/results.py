from pathlib import Path
from attrs import define
from typing import Dict, Optional
from xpmir.rankers import Scorer
from xpmir.evaluation import EvaluationsCollection


@define(kw_only=True)
class PaperResults:
    models: Dict[str, Scorer]
    """List of models with their identifier"""

    evaluations: EvaluationsCollection
    """The evaluation results"""

    tb_logs: Optional[Dict[str, Path]]
    """Tensorboard directory for each model"""
