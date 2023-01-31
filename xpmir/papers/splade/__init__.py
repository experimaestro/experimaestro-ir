from .. import Experiment

# Experiments
PAPERS = [
    Experiment(
        "splade_max", "spalde model by using the max aggregation", "experiment:cli"
    ),
    Experiment(
        "splade_doc",
        "splade model by using only the document encoder",
        "experiment_doc:cli",
    ),
    Experiment(
        "splade_DistilMSE",
        "splade model by using the hard negatives BM25 by using the distillation",
        "experiment_DistilMSE:cli",
    ),
]
