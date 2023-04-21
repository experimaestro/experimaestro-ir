import attr

try:
    from typing import dataclass_transform
except ImportError:
    from typing_extensions import dataclass_transform

from functools import cached_property as attrs_cached_property  # noqa: F401


@dataclass_transform(kw_only_default=True)
def configuration(*args, **kwargs):
    """Method to define keyword only dataclasses

    Configurations are keyword-only
    """

    return attr.define(*args, kw_only=True, slots=False, hash=True, eq=True, **kwargs)


@attr.define()
class Experiment:
    id: str
    """ID of the paper (command line)"""

    description: str
    """Description of the experiment"""

    cli: str
    """qualified name (relative to the module) for the CLI method"""
