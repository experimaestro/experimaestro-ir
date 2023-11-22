from pathlib import Path
from attrs import define
from typing import Dict, Optional, Union
from xpmir.rankers import Scorer
from xpmir.evaluation import EvaluationsCollection
from xpmir.text.huggingface import MLMEncoder


@define(kw_only=True)
class PaperResults:
    models: Dict[str, Union[Scorer, MLMEncoder]]
    """List of models with their identifier"""

    evaluations: EvaluationsCollection
    """The evaluation results"""

    tb_logs: Optional[Dict[str, Path]]
    """Tensorboard directory for each model"""
