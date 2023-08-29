from typing import Optional, Union
from pathlib import Path
from importlib import resources
from omegaconf import OmegaConf
from experimaestro.click import click


class OmegaConfParamType(click.Choice):
    def __init__(self, package, folder, names):
        super().__init__(names)
        self.package = package
        self.folder = folder

    def convert(self, value: str, param, ctx):
        # Load the YAML file
        if self.folder:
            return OmegaConf.load(self.folder / f"{value}.yaml")
        with resources.path(self.package, f"{value}.yaml") as path:
            return OmegaConf.load(path)


def omegaconf_argument(
    name: str,
    *,
    package: Optional[str] = None,
    folder: Optional[Union[str, Path]] = None,
    click_mode=click.argument,
    **kwargs,
):
    """Provides a choice of YAML configuration (file names)

    :param name: the name for the parameter
    :param folder: The folder containing YAML files (it can be either a Path or
        a string, and either a folder or a file within the folder)

    :param package: the module qualified name in which to search for yaml files
    """
    names = []  # list of available configurations

    YAML_SUFFIX = ".yaml"

    assert (
        folder is None or package is None
    ), "folder and package cannot be used at the same time"

    if folder is not None:
        if isinstance(folder, str):
            folder = Path(folder)
        if folder.is_file():
            folder = folder.parent

        for config_path in folder.glob(f"*{YAML_SUFFIX}"):
            names.append(config_path.name[: -len(YAML_SUFFIX)])

    if package is not None:
        for item in resources.contents(package):
            if resources.is_resource(package, item) and item.endswith(YAML_SUFFIX):
                names.append(item[: -len(YAML_SUFFIX)])

    def handler(method):
        return click_mode(
            name, type=OmegaConfParamType(package, folder, names), **kwargs
        )(method)

    return handler
