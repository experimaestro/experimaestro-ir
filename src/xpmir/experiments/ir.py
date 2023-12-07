import click
from xpmir.papers.results import PaperResults
from xpmir.experiments import Experiment
from experimaestro import RunMode, experiment


class ir_experiment(Experiment):
    """Wraps an experiment into an IR experiment

    1. Upload to github (if requested)
    2.

    :param func: The function to be wrapped
    """

    def __init__(self, *args):
        if len(args) == 1:
            self.func = args[0]

    def __call__(self, func):
        self.func = func
        return self

    def run(self, extra_args, xp: experiment, *args, **kwargs):
        @click.option("--upload_to_hub", type=str)
        @click.command()
        def cli(upload_to_hub: str):
            results = self.func(xp, *args, **kwargs)

            if isinstance(results, PaperResults) and xp.run_mode == RunMode.NORMAL:
                if upload_to_hub is not None:
                    upload_to_hub.send_scorer(
                        results.models,
                        evaluations=results.evaluations,
                        tb_logs=results.tb_logs,
                    )

                print(results.evaluations.to_dataframe())  # noqa: T201

        return cli(extra_args, standalone_mode=False)
