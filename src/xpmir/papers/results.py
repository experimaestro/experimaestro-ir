from pathlib import Path
from attrs import define
from typing import Dict, Optional, Union
from xpmir.rankers import Scorer
from xpmir.evaluation import EvaluationsCollection
from xpmir.text.huggingface.base import HFMaskedLanguageModel
from xpm_torch.results import TrainingResults


@define(kw_only=True)
class PaperResults:
    """Results from an IR paper experiment."""

    models: Dict[str, Union[Scorer, HFMaskedLanguageModel]]
    """List of models with their identifier"""

    evaluations: EvaluationsCollection
    """The evaluation results"""

    tb_logs: Optional[Dict[str, Path]]
    """Tensorboard directory for each model"""

    def to_training_results(self) -> TrainingResults:
        """Convert to a serializable TrainingResults for xp.save()."""
        return TrainingResults.C(models=self.models, tb_logs=self.tb_logs)
