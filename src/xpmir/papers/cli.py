# Starts experiments from command line

import inspect
import io
import json
import logging
import pkgutil
import sys
from functools import reduce
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import click
import docstring_parser
import omegaconf
from experimaestro import LauncherRegistry, Config, RunMode, experiment
from experimaestro.settings import get_workspace
from omegaconf import OmegaConf, SCMode
from termcolor import cprint

import xpmir.papers as papers
from xpmir.configuration import omegaconf_argument
from xpmir.evaluation import EvaluationsCollection
from xpmir.learning.optim import TensorboardService
from xpmir.models import XPMIRHFHub
from xpmir.papers.helpers import PaperExperiment
from xpmir.papers.results import PaperResults


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
        try:
            experiment = self.id2experiment[cmd_name]
        except KeyError:
            logging.error(
                "Configuration %s not found among %s",
                cmd_name,
                " ".join(self.id2experiment.keys()),
            )
            raise
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


class UploadToHub:
    def __init__(self, model_id: Optional[str], doc: docstring_parser.Docstring):
        self.model_id = model_id
        self.doc = doc

    def send_scorer(
        self,
        models: Dict[str, Config],
        *,
        evaluations: Optional[EvaluationsCollection] = None,
        tb_logs: Dict[str, Path],
        add_des: str = "",
    ):
        """Upload the scorer(s) to the HuggingFace Hub

        :param models: The models to upload, each with a key
        :param tb_logs: The tensorboard logs
        :param evaluations: Models evaluations, defaults to None
        :param add_des: Extra documentation, defaults to ""
        """
        if self.model_id is None:
            return

        out = io.StringIO()
        out.write(
            """---
library_name: xpmir
---
"""
        )

        out.write(f"{self.doc}\n\n")
        out.write(f"{add_des}\n\n")

        out.write("\n## Using the model")
        out.write(
            f"""
The model can be loaded with [experimaestro
IR](https://experimaestro-ir.readthedocs.io/en/latest/)

```py from xpmir.models import AutoModel
from xpmir.models import AutoModel

# Model that can be re-used in experiments
model, init_tasks = AutoModel.load_from_hf_hub("{self.model_id}")

# Use this if you want to actually use the model
model = AutoModel.load_from_hf_hub("{self.model_id}", as_instance=True)
model.rsv("walgreens store sales average", "The average Walgreens salary ranges...")
```
"""
        )

        assert len(models) == 1, "Cannot deal with more than one variant"
        ((key, model),) = list(models.items())

        if evaluations is not None:
            out.write("\n## Results\n\n")
            evaluations.output_model_results(key, file=out)

        readme_md = out.getvalue()

        logging.info("Uploading to HuggingFace Hub")
        XPMIRHFHub(model, readme=readme_md, tb_logs=tb_logs).push_to_hub(
            repo_id=self.model_id, config={}
        )


def paper_command(
    package: Optional[str] = None,
    folder: Optional[Union[str, Path]] = None,
    schema=None,
    tensorboard_service=False,
):
    """General command line decorator for an XPM-IR experiment

    This annotation adds a set of arguments for the

    HuggingFace upload: the documentation comes from the docstring

    :param tensorboard_service: If true, register a tensorboard service and transmits it
        as ``tensorboard_service`` to function.

    .. seealso:: :py:func:`omegaconf_argument` for `folder` and `package` documentation
    """

    omegaconf_schema = None
    if schema is not None:
        omegaconf_schema = OmegaConf.structured(schema())

    def _decorate(fn):
        decorators = [
            click.command(),
            click.option("--debug", is_flag=True, help="Print debug information"),
            click.option("--show", is_flag=True, help="Print configuration and exits"),
            click.option(
                "--env",
                help="Define one environment variable",
                type=(str, str),
                multiple=True,
            ),
            click.option(
                "--host",
                type=str,
                default=None,
                help="Server hostname (default to localhost,"
                " not suitable if your jobs are remote)",
            ),
            click.option(
                "--run-mode",
                type=click.Choice(RunMode),
                default=RunMode.NORMAL,
                help="Sets the run mode",
            ),
            click.option(
                "--xpm-config-dir",
                type=Path,
                default=None,
                help="Path for the experimaestro config directory "
                "(if not specified, use $HOME/.config/experimaestro)",
            ),
            click.option(
                "--port",
                type=int,
                default=None,
                help="Port for monitoring (can be defined in the settings.yaml file)",
            ),
            click.option(
                "--upload-to-hub",
                required=False,
                type=str,
                default=None,
                help="Upload the model to Hugging Face Hub with the given identifier",
            ),
            click.option(
                "--workdir",
                type=str,
                default=None,
                help="Working directory - if None, uses the default XPM "
                "working directory",
            ),
            omegaconf_argument(
                "--configuration",
                package=package,
                folder=folder,
                click_mode=click.option,
                required=True,
            ),
            click.argument("args", nargs=-1, type=click.UNPROCESSED),
        ]

        def cli(
            show,
            debug,
            configuration,
            workdir: Path,
            host: Optional[str],
            port: Optional[int],
            xpm_config_dir: Optional[Path],
            env: List[Tuple[str, str]],
            args,
            run_mode: RunMode,
            upload_to_hub: Optional[str],
            **kwargs,
        ):
            nonlocal omegaconf_schema
            assert schema is None or omegaconf_schema is not None

            logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
            logging.getLogger("xpm.hash").setLevel(logging.INFO)
            conf_args = OmegaConf.from_dotlist(args)

            if xpm_config_dir is not None:
                assert xpm_config_dir.is_dir()
                LauncherRegistry.set_config_dir(xpm_config_dir)

            configuration: PaperExperiment = OmegaConf.merge(configuration, conf_args)
            if omegaconf_schema is not None:
                try:
                    configuration: PaperExperiment = OmegaConf.merge(
                        omegaconf_schema, configuration
                    )
                except omegaconf.errors.ConfigKeyError as e:
                    cprint(f"Error in configuration:\n\n{e}", "red", file=sys.stderr)
                    sys.exit(1)

            if show:
                print(json.dumps(OmegaConf.to_container(configuration)))  # noqa: T201
                sys.exit(0)

            configuration = OmegaConf.to_container(
                configuration, structured_config_mode=SCMode.INSTANTIATE
            )
            parameters = inspect.signature(fn).parameters

            if upload_to_hub is not None:
                if configuration.title == "" and configuration.description == "":
                    doc = docstring_parser.parse(fn.__doc__)
                else:
                    doc = f"# {configuration.title}\n{configuration.description}"
                upload_to_hub = UploadToHub(upload_to_hub, doc)
                kwargs["upload_to_hub"] = upload_to_hub

            kwargs = {**kwargs, "debug": debug, "run_mode": run_mode}

            kwargs = {key: value for key, value in kwargs.items() if key in parameters}

            # Run the experiment
            if run_mode == RunMode.NORMAL:
                logging.info("Starting experimaestro server (%s:%s)", host, port)

            if workdir is None or not Path(workdir).is_dir():
                workdir = get_workspace(workdir).path.expanduser().resolve()
                logging.info("Using working directory %s", workdir)

            with experiment(
                workdir, configuration.id, host=host, port=port, run_mode=run_mode
            ) as xp:
                if tensorboard_service:
                    kwargs["tensorboard_service"] = xp.add_service(
                        TensorboardService(xp.resultspath / "runs")
                    )

                for key, value in env:
                    xp.setenv(key, value)

                results = fn(xp, configuration, **kwargs)
                xp.wait()

                if isinstance(results, PaperResults) and run_mode == RunMode.NORMAL:
                    if upload_to_hub is not None and "upload_to_hub" not in parameters:
                        upload_to_hub.send_scorer(
                            results.models,
                            evaluations=results.evaluations,
                            tb_logs=results.tb_logs,
                        )

                    print(results.evaluations.to_dataframe())  # noqa: T201
                return results

        cli.__doc__ = fn.__doc__
        cmd = reduce(lambda fn, decorator: decorator(fn), decorators, cli)
        return cmd

    return _decorate


papers_cli = PapersCli(help="Runs an experiment from a paper")
