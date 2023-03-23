from attrs import define
from omegaconf import MISSING


@define(kw_only=True)
class PaperExperiment:
    id: str = MISSING
    """The experiment ID

    The ID should be unique, and might be used when exporting the model e.g. to
    HuggingFace
    """

    title: str = ""
    """The model title

    The title is used to generate report
    """

    description: str = ""
    """A description of the model

    This description is used to generate reports
    """
