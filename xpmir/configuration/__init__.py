from importlib import resources
from omegaconf import OmegaConf
from experimaestro.click import click

class OmegaConfParamType(click.Choice):
    # name = "OmegaConf"
    def __init__(self, package, names):
        super().__init__(self, names)
        self.package = package
        
    def convert(self, value: str, param, ctx):
        # Load the YAML file
        # value = super().convert(value, param, ctx)
        value = value + '.yaml'
        self.path = resources.path(self.package, value)
        conf = OmegaConf.load(self.path)
        return conf


def omegaconf_argument(name: str, package: str=None):
    ''' name: the name for the parameter,
    package: the path to stock the configuration files
    '''
    names = [] # list of available configurations

    for item in resources.contents(package):
        if resources.is_resource(package, item) and item.endswith('.yaml'):
            names.append(item.removesuffix(".yaml"))
    
    def handler(method):
        return click.argument(name, type=OmegaConfParamType(package, names))(method)
    return handler