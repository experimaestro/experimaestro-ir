# See documentation on https://datamaestro.readthedocs.io

from datamaestro.definitions import datatasks, datatags, dataset
from datamaestro_text.data.ir.huggingface import HuggingFacePairwiseSampleDataset
from datamaestro.download.huggingface import hf_download


@datatags("information retrieval", "hard negatives", "msmarco")
@datatasks("learning to rank")
@hf_download(
    "dataset",
    "sentence-transformers/msmarco-hard-negatives",
    data_files="msmarco-hard-negatives-msmarco.jsonl.gz",
    split="train",
)
@dataset(
    HuggingFacePairwiseSampleDataset,
    url="https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives",
)
def ensemble(dataset):
    """Hard negatives mined from a set of models"""
    return dataset
