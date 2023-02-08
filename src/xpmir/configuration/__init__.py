from importlib import resources
from omegaconf import OmegaConf
from experimaestro.click import click


class OmegaConfParamType(click.Choice):
    def __init__(self, package, names):
        super().__init__(names)
        self.package = package

    def convert(self, value: str, param, ctx):
        # Load the YAML file
        with resources.path(self.package, f"{value}.yaml") as path:
            return OmegaConf.load(path)


def omegaconf_argument(
    name: str, package: str = None, click_mode=click.argument, **kwargs
):
    """Provides a choice of YAML configuration (file names)

    :param name: the name for the parameter,

    :param package: the module qualified name in which to search for yaml files
    """
    names = []  # list of available configurations

    for item in resources.contents(package):
        YAML_SUFFIX = ".yaml"
        if resources.is_resource(package, item) and item.endswith(YAML_SUFFIX):
            names.append(item[: -len(YAML_SUFFIX)])

    def handler(method):
        return click_mode(name, type=OmegaConfParamType(package, names), **kwargs)(
            method
        )

    return handler
