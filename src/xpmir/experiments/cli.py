import sys
from typing import List, Tuple
from experimaestro import LauncherRegistry, Config, RunMode, experiment
from pathlib import Path
import click
import yaml


def load(yaml_file: Path):
    """Loads a YAML file, and parents one if they exist"""
    if not yaml_file.exists() and yaml_file.suffix != ".yaml":
        yaml_file = yaml_file.with_suffix(".yaml")

    with yaml_file.open("rt") as fp:
        _data = yaml.full_load(fp)
    data = [_data]
    if parent := _data.get("parent", None):
        data.extend(load(yaml_file.parent / parent))
        del _data["parent"]

    return data


@click.option("--debug", is_flag=True, help="Print debug information")
@click.option("--show", is_flag=True, help="Print configuration and exits")
@click.option(
    "--env",
    help="Define one environment variable",
    type=(str, str),
    multiple=True,
)
@click.option(
    "--host",
    type=str,
    default=None,
    help="Server hostname (default to localhost,"
    " not suitable if your jobs are remote)",
)
@click.option(
    "--run-mode",
    type=click.Choice(RunMode),
    default=RunMode.NORMAL,
    help="Sets the run mode",
)
@click.option(
    "--xpm-config-dir",
    type=Path,
    default=None,
    help="Path for the experimaestro config directory "
    "(if not specified, use $HOME/.config/experimaestro)",
)
@click.option(
    "--port",
    type=int,
    default=None,
    help="Port for monitoring (can be defined in the settings.yaml file)",
)
@click.option(
    "--file", "xp_file", help="The file containing the main experimental code"
)
@click.argument("yaml_file", metavar="YAML file", type=str)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
@click.command()
def experiments_cli(
    yaml_file: List[str],
    xp_file: str,
    host: str,
    port: int,
    xpm_config_dir: Path,
    env: List[Tuple[str, str]],
    run_mode: RunMode,
    args: List[str],
    show: bool,
    debug: bool,
):
    """Run an experiment"""

    # --- Loads the YAML
    yamls = load(Path(yaml_file))

    # --- Get the XP file
    if xp_file is None:
        for data in yamls:
            if xp_file := data.get("file"):
                break

        if xp_file is None:
            raise ValueError("No experiment file given")

    # --- Loads the XP file
    xp_file = Path(xp_file)
    if not xp_file.exists() and xp_file.suffix != ".py":
        xp_file = xp_file.with_suffix(".py")
    xp_file = Path(yaml_file).parent / xp_file

    with open(xp_file, "r") as f:
        source = f.read()
    if sys.version_info < (3, 9):
        the__file__ = str(xp_file)
    else:
        the__file__ = str(xp_file.absolute())

    _globals = {}
    _locals = {}
    code = compile(source, filename=the__file__, mode="exec")
    exec(code, _globals, _locals)

    # --- ... and runs it
