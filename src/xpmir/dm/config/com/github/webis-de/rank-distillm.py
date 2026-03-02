# See documentation on https://datamaestro.readthedocs.io
from hashlib import md5
from datamaestro.definitions import dataset
from datamaestro.download.single import filedownloader
from datamaestro.utils import HashCheck
from xpmir.letor.distillation.samplers import ListwiseDistillationSamplesTSV, ListwiseDistillationSamplesTSVWithAnnotations
from datamaestro import prepare_dataset

@filedownloader(
    "bm25__msmarco_passage_train_judged.run",
    "https://zenodo.org/records/12528410/files/"
    "__bm25__msmarco-passage-train-judged.run?download=1",
    checker=HashCheck("835372b2ab4d20acf10addeae526c559", md5),
)
@dataset(
    ListwiseDistillationSamplesTSVWithAnnotations,
    url="https://github.com/webis-de/rank-distillm",
)
def msmarco_bm25_annotated(bm25__msmarco_passage_train_judged):
    """Top 500 passages for all queries that have at least one relevance judgement 
    in the MS MARCO training query set retrieved by BM25.
    """
    return {
        "qrels": prepare_dataset("irds.msmarco-passage.train.qrels"),
        "top_k": 500,
        "sampling_k": 8,
        "with_docid": True,
        "with_queryid": True,
        "path": bm25__msmarco_passage_train_judged,
    }

@filedownloader(
    "colbert__msmarco_passage_train_judged.run",
    "https://zenodo.org/records/12528410/files/"
    "__colbert__msmarco-passage-train-judged.run?download=1",
    checker=HashCheck("6ed152027f7270f32fcbfaaa6def951e", md5),
)
@dataset(
    ListwiseDistillationSamplesTSVWithAnnotations,
    url="https://github.com/webis-de/rank-distillm",
)
def msmarco_colbertv2_annotated(colbert__msmarco_passage_train_judged):
    """Top 500 passages retrieved by ColBERTv2 for all queries in the MS MARCO training query set.
    WARNING: not all 500 docs necessarily contains relevant documents. 
    """
    return {
        "qrels": prepare_dataset("irds.msmarco-passage.train.qrels"),
        "top_k": 500,
        "with_docid": True,
        "with_queryid": True,
        "path": colbert__msmarco_passage_train_judged,
    }


@filedownloader(
    "rankzephyr_bm25_10000_sampled_100__msmarco_passage_train_judged.run",
    "https://zenodo.org/records/12528410/files/"
    "__rankzephyr-bm25-10000-sampled-100__msmarco-passage-train-judged.run?download=1",
    checker=HashCheck("05e3137ea3526671e1565cc90f9a2c8a ", md5),
)
@dataset(
    ListwiseDistillationSamplesTSV,
    url="https://github.com/webis-de/rank-distillm",
)
def rankzephyr_bm25_10000_sampled_100_annotated(rankzephyr_bm25_10000_sampled_100__msmarco_passage_train_judged):
    """Top 100 passages retrieved by BM25 for 10k queries sampled from the MSMARCO training set.

    All passages are then reranked using RankZephyr and can be used for distillation.
    """
    return {
        "top_k": 100,
        "with_docid": True,
        "with_queryid": True,
        "path": rankzephyr_bm25_10000_sampled_100__msmarco_passage_train_judged,
    }


@filedownloader(
    "rankzephyr_colbert_10000_sampled_100__msmarco_passage_train_judged.run",
    "https://zenodo.org/records/12528410/files/"
    "__rankzephyr-colbert-10000-sampled-100__msmarco-passage-train-judged.run?download=1",
    checker=HashCheck("49f8dbf2c1ee7a2ca1fe517eda528af6  ", md5),
)
@dataset(
    ListwiseDistillationSamplesTSV,
    url="https://github.com/webis-de/rank-distillm",
)
def rankzephyr_colbert_10000_sampled_100_annotated(rankzephyr_colbert_10000_sampled_100__msmarco_passage_train_judged):
    """Top 100 passages retrieved by ColBERT for 10k queries sampled from the MSMARCO training set.

    All passages are then reranked using RankZephyr and can be used for distillation.
    """
    return {
        "top_k": 100,
        "with_docid": True,
        "with_queryid": True,
        "path": rankzephyr_colbert_10000_sampled_100__msmarco_passage_train_judged,
    }


@filedownloader(
    "rankzephyr_colbert_10000_sampled_50__msmarco_passage_train_judged.run",
    "https://zenodo.org/records/12528410/files/"
    "__rankzephyr-colbert-10000-sampled-50__msmarco-passage-train-judged.run?download=1",
    checker=HashCheck("c37b78874d4893a00566ab40aa453c56", md5),
)
@dataset(
    ListwiseDistillationSamplesTSV,
    url="https://github.com/webis-de/rank-distillm",
)
def rankzephyr_colbert_10000_sampled_50_annotated(rankzephyr_colbert_10000_sampled_50__msmarco_passage_train_judged):
    """Top 50 passages retrieved by ColBERT for 10k queries sampled from the MSMARCO training set.

    All passages are then reranked using RankZephyr and can be used for distillation.
    """
    return {
        "top_k": 50,
        "with_docid": True,
        "with_queryid": True,
        "path": rankzephyr_colbert_10000_sampled_50__msmarco_passage_train_judged,
    }

@filedownloader(
    "rankzephyr_colbert_10000_sampled_10__msmarco_passage_train_judged.run",
    "https://zenodo.org/records/12528410/files/"
    "__rankzephyr-colbert-10000-sampled-10__msmarco-passage-train-judged.run?download=1",
    checker=HashCheck("619bc815bd133bdca44d6331b241d39a  ", md5),
)
@dataset(
    ListwiseDistillationSamplesTSV,
    url="https://github.com/webis-de/rank-distillm",
)
def rankzephyr_colbert_10000_sampled_10_annotated(rankzephyr_colbert_10000_sampled_10__msmarco_passage_train_judged):
    """Top 10 passages retrieved by ColBERT for 10k queries sampled from the MSMARCO training set.

    All passages are then reranked using RankZephyr and can be used for distillation.
    """
    return {
        "top_k": 10,
        "with_docid": True,
        "with_queryid": True,
        "path": rankzephyr_colbert_10000_sampled_10__msmarco_passage_train_judged,
    }