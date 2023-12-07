from .cli import experiments_cli  # noqa: F401
from typing import Callable


class Experiment:
    func: Callable
