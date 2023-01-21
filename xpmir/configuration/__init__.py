from importlib import resources
from omegaconf import OmegaConf
from experimaestro.click import click


class OmegaConfParamType(click.Choice):
    def __init__(self, package, names):
        super().__init__(names)
        self.package = package

    def convert(self, value: str, param, ctx):
        # Load the YAML file
        self.path = resources.path(self.package, f"{value}.yaml")
        return OmegaConf.load(self.path)


def omegaconf_argument(name: str, package: str = None):
    """Provides a choice of YAML configuration (file names)

    :param name: the name for the parameter,

    :param package: the module qualified name in which to search for yaml files
    """
    names = []  # list of available configurations

    for item in resources.contents(package):
        if resources.is_resource(package, item) and item.endswith(".yaml"):
            names.append(item.removesuffix(".yaml"))

    def handler(method):
        return click.argument(name, type=OmegaConfParamType(package, names))(method)

    return handler
