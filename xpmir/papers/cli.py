# Starts experiments from command line

from typing import List
from pathlib import Path
import pkgutil
from typing import Optional
import click
from importlib import import_module
import xpmir.papers as papers


class PaperCli(click.MultiCommand):
    def __init__(self, papers: List[papers.Paper], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.papers = papers

    def list_commands(self, ctx: click.Context):
        return [paper.id for paper in self.papers]

    def get_command(self, ctx: click.Context, cmd_name: str) -> Optional[click.Command]:
        return


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

            papers = mod.PAPERS  # type: List[papers.Paper]
            return PaperCli(papers)
        except AttributeError:
            pass

        return


papers_cli = PapersCli(help="Runs an experiment from a paper")
