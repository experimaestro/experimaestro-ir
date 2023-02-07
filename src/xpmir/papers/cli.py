# Starts experiments from command line

from functools import reduce
import inspect
import io
import logging
import sys
from typing import Dict, List
from pathlib import Path
import pkgutil
from typing import Optional
import click
from importlib import import_module
import docstring_parser
from attrs import define

from experimaestro import experiment
from omegaconf import OmegaConf, MISSING
from xpmir.configuration import omegaconf_argument
from xpmir.evaluation import EvaluationsCollection
import xpmir.papers as papers
from xpmir.models import XPMIRHFHub
from xpmir.rankers import Scorer


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


class UploadToHub:
    def __init__(self, model_id: Optional[str], doc: docstring_parser.Docstring):
        self.model_id = model_id
        self.doc = doc

    def send_scorer(
        self,
        models: Dict[str, Scorer],
        *,
        evaluations: Optional[EvaluationsCollection] = None,
    ):
        if self.model_id is None:
            return

        out = io.StringIO()
        out.write(
            f"""---
library_name: xpmir
---
"""
        )

        out.write(f"# {self.doc.short_description}\n\n")
        out.write(f"{self.doc.long_description}\n")

        out.write("\n## Using the model")
        out.write(
            f""")
The model can be loaded with [experimaestro IR](https://experimaestro-ir.readthedocs.io/en/latest/)

```py
from xpmir.models import AutoModel

# Model that can be re-used in experiments
model = AutoModel.load_from_hf_hub("{self.model_id}")

# Use this if you want to actually use the model
model = AutoModel.load_from_hf_hub("{self.model_id}", as_instance=True)
model.initialize(None)
model.rsv("walgreens store sales average", "The average Walgreens salary ranges from approximately $15,000 per year for Customer Service Associate / Cashier to $179,900 per year for District Manager...")
```
"""
        )

        assert len(models) == 1, f"Cannot deal with more than one variant"
        ((key, model),) = list(models.items())

        if evaluations is not None:
            out.write("\n## Results\n")
            evaluations.output_model_results(key, file=out)

        readme_md = out.getvalue()

        logging.info("Uploading to HuggingFace Hub")
        XPMIRHFHub(model, readme=readme_md).push_to_hub(
            repo_id=self.model_id, config={}
        )


@define(kw_only=True)
class PaperExperiment:
    id: str = MISSING
    """The experiment ID"""


def paper_command(package=None, schema=None):
    """General command line decorator for an XPM-IR experiment"""

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
                help="Server hostname (default to localhost, not suitable if your jobs are remote)",
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
            click.argument("workdir", type=Path),
            omegaconf_argument("configuration", package=package),
            click.argument("args", nargs=-1, type=click.UNPROCESSED),
        ]

        def cli(
            show,
            debug,
            configuration,
            workdir,
            host,
            port,
            args,
            env,
            upload_to_hub: Optional[str],
            **kwargs,
        ):
            nonlocal omegaconf_schema
            assert schema is None or omegaconf_schema is not None

            logging.getLogger().setLevel(logging.DEBUG if debug else logging.INFO)
            conf_args = OmegaConf.from_dotlist(args)

            configuration: PaperExperiment = OmegaConf.merge(configuration, conf_args)
            if omegaconf_schema is not None:
                configuration: PaperExperiment = OmegaConf.merge(
                    omegaconf_schema, configuration
                )

            if show:
                # flake8: noqa: T201
                print(configuration)
                sys.exit(0)

            parameters = inspect.signature(fn).parameters

            doc = docstring_parser.parse(fn.__doc__)

            if "documentation" in parameters:
                kwargs[
                    "documentation"
                ] = f"{doc.short_description}\n\n{doc.long_description}"

            if "upload_to_hub" in parameters:
                kwargs["upload_to_hub"] = UploadToHub(upload_to_hub, doc)

            if "debug" in parameters:
                kwargs["debug"] = debug

            # Run the experiment
            with experiment(workdir, configuration.id, host=host, port=port) as xp:

                for key, value in env:
                    xp.setenv(key, value)

                return fn(xp, configuration, **kwargs)

        cli.__doc__ = fn.__doc__
        cmd = reduce(lambda fn, decorator: decorator(fn), decorators, cli)
        return cmd

    return _decorate


papers_cli = PapersCli(help="Runs an experiment from a paper")
