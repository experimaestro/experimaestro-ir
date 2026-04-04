from pathlib import Path
from attrs import define
from typing import Any, Dict, Optional
from xpmir.evaluation import EvaluationsCollection
from xpm_torch.module import ModuleLoader
from xpm_torch.results import TrainingResults


@define(kw_only=True)
class PaperResults:
    """Results from an IR paper experiment."""

    models: Dict[str, Any]
    """Model loaders keyed by identifier (ValidationModuleLoader, ModuleLoader, etc.)"""

    evaluations: EvaluationsCollection
    """The evaluation results"""

    tb_logs: Optional[Dict[str, Path]]
    """Tensorboard directory for each model"""

    def to_training_results(self) -> TrainingResults:
        """Convert to a serializable TrainingResults for xp.save().

        Unwraps ValidationModuleLoader/CheckpointModuleLoader wrappers
        to get the inner ModuleLoader, since TrainingResults expects
        plain ModuleLoaders.
        """
        loaders = {}
        for key, model in self.models.items():
            if isinstance(model, ModuleLoader):
                loaders[key] = model
            elif hasattr(model, "loader") and isinstance(model.loader, ModuleLoader):
                loaders[key] = model.loader
            else:
                loaders[key] = model
        return TrainingResults.C(models=loaders, tb_logs=self.tb_logs)
