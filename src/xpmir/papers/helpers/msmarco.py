# Utility functions for MS-Marco experiments

from attrs import Factory
import logging

from . import NeuralIRExperiment, configuration
from xpmir.papers.helpers.samplers import ValidationSample

logging.basicConfig(level=logging.INFO)


@configuration()
class RerankerMSMarcoV1Configuration(NeuralIRExperiment):
    """Configuration for rerankers"""

    validation: ValidationSample = Factory(ValidationSample)


@configuration()
class DualMSMarcoV1Configuration(NeuralIRExperiment):
    """Configuration for sparse retriever"""

    validation: ValidationSample = Factory(ValidationSample)


@configuration()
class MLMMSMarcoV1Configuration(NeuralIRExperiment):
    """Configuration for sparse retriever"""

    validation: ValidationSample = Factory(ValidationSample)
