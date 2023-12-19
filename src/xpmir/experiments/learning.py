import inspect
from functools import cached_property

from xpmir.experiments import ExperimentHelper
from xpmir.learning.optim import TensorboardService


class LearningExperimentHelper(ExperimentHelper):
    @cached_property
    def tensorboard_service(self) -> TensorboardService:
        """Returns a tensorboard service"""
        return self.xp.add_service(TensorboardService(self.xp.resultspath / "runs"))

    # TODO: remove when in experimaestro
    @classmethod
    def decorator(cls, *args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and inspect.isfunction(args[0]):
            return cls(callable)

        def wrapper(callable):
            return cls(callable)

        return wrapper


learning_experiment = LearningExperimentHelper.decorator
"""Wraps an experiment into an experiment where a model is learned

Provides:

1. Tensorboard service
"""
