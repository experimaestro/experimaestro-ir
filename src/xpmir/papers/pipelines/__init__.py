from attrs import define
from omegaconf import MISSING


@define(kw_only=True)
class PaperExperiment:
    id: str = MISSING
    """The experiment ID"""

    title: str = ""
    """The model title"""

    description: str = ""
    """A description of the model"""
