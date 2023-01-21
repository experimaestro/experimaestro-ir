# Starts experiments from command line

from functools import reduce
import logging
import sys
from typing import List
from pathlib import Path
import pkgutil
from typing import Optional
import click
from importlib import import_module

from omegaconf import OmegaConf
from xpmir.configuration import omegaconf_argument
import xpmir.papers as papers


class ExperimentsCli(click.MultiCommand):
    def __init__(
        self, pkg_name: str, experiments: List[papers.Experiment], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pkg_name = pkg_name
        self.id2experiment = {experiment.id: experiment for experiment in experiments}
        self.experiments = experiments

    def list_commands(self, ctx: click.Context):
        return [experiment.id for experiment in self.experiments]

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        experiment = self.id2experiment[cmd_name]
        sub_package, name = experiment.cli.split(":")
        module = import_module(f"{self.pkg_name}.{sub_package}")
        return getattr(module, name)


class PapersCli(click.MultiCommand):
    def list_commands(self, ctx):
        path = str(Path(papers.__file__).parent)
        names = []
        for pkg in pkgutil.walk_packages([path]):
            names.append(pkg.name)
        return names

    def get_command(self, ctx, name):
        pkg_name = f"{__package__}.{name}"
        try:
            mod = import_module(pkg_name)

            papers = mod.PAPERS  # type: List[papers.Experiment]
            return ExperimentsCli(pkg_name, papers)
        except AttributeError:
            pass

        return


def paper_command(package=None):
    """General command line decorator for an XPM-IR experiment"""

    def _decorate(fn):
        decorators = [
            click.command(),
            click.option("--debug", is_flag=True, help="Print debug information"),
            click.option("--show", is_flag=True, help="Print configuration and exits"),
            click.option(
                "--host",
                type=str,
                default=None,
                help="Server hostname (default to localhost, not suitable if your jobs are remote)",
            ),
            click.option(
                "--port",
                type=int,
                default=12345,
                help="Port for monitoring (default 12345)",
            ),
            click.argument("workdir", type=Path),
            omegaconf_argument("configuration", package=package),
            click.argument("args", nargs=-1, type=click.UNPROCESSED),
        ]

        def cli(debug, configuration, host, port, workdir, args, show):
            logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
            conf_args = OmegaConf.from_dotlist(args)
            configuration = OmegaConf.merge(configuration, conf_args)

            if show:
                # flake8: noqa: T201
                print(configuration)
                sys.exit(0)

            fn(debug, configuration, host, port, workdir)

        return reduce(lambda fn, decorator: decorator(fn), decorators, cli)

    return _decorate


papers_cli = PapersCli(help="Runs an experiment from a paper")
