import attr


@attr.define()
class Experiment:
    id: str
    """ID of the paper (command line)"""

    description: str
    """Description of the experiment"""

    cli: str
    """qualified name (relative to the module) for the CLI method"""
