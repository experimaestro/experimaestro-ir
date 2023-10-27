from .. import Experiment

# Experiments
PAPERS = [
    Experiment(
        "msmarco", "mono-BERT trained on MS-Marco passages (v1)", "experiment:cli"
    ),
    Experiment(
        "proba_tab", "proba_tab for generative retrival", "experiment_proba_tab:cli"
    ),
]
