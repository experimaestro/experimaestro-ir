from .. import Experiment

# Experiments
PAPERS = [
    Experiment(
        "msmarco", "mono-BERT trained on MS-Marco passages (v1)", "experiment:cli"
    ),
    Experiment(
        "finetune",
        "Finetune monoBERT on a specific dataset (temporary)",
        "finetune:cli",
    ),
]
