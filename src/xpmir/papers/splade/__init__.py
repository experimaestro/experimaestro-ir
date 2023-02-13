from .. import Experiment

# Experiments
PAPERS = [
    Experiment(
        "spladeV2",
        "spaldeV2 models, could be run with different config",
        "experiment:cli",
    ),
    Experiment(
        "splade_DistilMSE",
        "splade model by using the hard negatives by using the distillation",
        "experiment_DistilMSE:cli",
    ),
]
