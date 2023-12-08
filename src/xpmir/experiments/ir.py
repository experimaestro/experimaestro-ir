from typing import Any, List
import click
import inspect

from xpmir.papers.results import PaperResults
from xpmir.experiments import ExperimentHelper
from experimaestro import RunMode


class IRExperimentHelper(ExperimentHelper):
    def run(self, extra_args: List[str], config: Any):
        @click.option("--upload_to_hub", type=str)
        @click.command()
        def cli(upload_to_hub: str):
            results = self.callable(self, config)
            self.xp.wait()

            if isinstance(results, PaperResults) and self.xp.run_mode == RunMode.NORMAL:
                if upload_to_hub is not None:
                    upload_to_hub.send_scorer(
                        results.models,
                        evaluations=results.evaluations,
                        tb_logs=results.tb_logs,
                    )

                print(results.evaluations.to_dataframe())  # noqa: T201

        return cli(extra_args, standalone_mode=False)


def ir_experiment(*args, **kwargs):
    """Wraps an experiment into an IR experiment

    1. Upload to github (if requested)
    2.

    :param func: The function to be wrapped
    """
    if len(args) == 1 and len(kwargs) == 0 and inspect.isfunction(args[0]):
        return IRExperimentHelper(callable)

    def wrapper(callable):
        return IRExperimentHelper(callable)

    return wrapper
