# See documentation on https://datamaestro.readthedocs.io
from hashlib import md5
from datamaestro.definitions import dataset
from datamaestro.download.single import filedownloader
from datamaestro.utils import HashCheck
from xpmir.letor.distillation.samplers import PairwiseDistillationSamplesTSV


@filedownloader(
    "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv",
    "https://zenodo.org/record/4068216/files/"
    "bert_cat_ensemble_msmarcopassage_train_scores_ids.tsv?download=1",
    checker=HashCheck("4d99696386f96a7f1631076bcc53ac3c", md5),
)
@dataset(
    PairwiseDistillationSamplesTSV,
    url="https://github.com/sebastian-hofstaetter/neural-ranking-kd",
)
def msmarco_ensemble_teacher(bert_cat_ensemble_msmarcopassage_train_scores_ids):
    """Training files without the text content instead using the ids from MSMARCO

    The teacher files (using the data from "Train Triples Small" with ~40
    million triples) with the format pos_score neg_score query_id pos_passage_id
    neg_passage_id (with tab separation)
    """
    return {
        "with_docid": True,
        "with_queryid": True,
        "path": bert_cat_ensemble_msmarcopassage_train_scores_ids,
    }


@filedownloader(
    "bertbase_cat_msmarcopassage_train_scores_ids.tsv",
    "https://zenodo.org/record/4068216/files/"
    "bertbase_cat_msmarcopassage_train_scores_ids.tsv?download=1",
    checker=HashCheck("a2575af08a19b47c2041e67c9efcd917", md5),
)
@dataset(
    PairwiseDistillationSamplesTSV,
    url="https://github.com/sebastian-hofstaetter/neural-ranking-kd",
)
def msmarco_bert_teacher(bertbase_cat_msmarcopassage_train_scores_ids):
    """Training files without the text content instead using the ids from MSMARCO

    The teacher files (using the data from "Train Triples Small" with ~40
    million triples) with the format pos_score neg_score query_id pos_passage_id
    neg_passage_id (with tab separation)
    """
    return {
        "with_docid": True,
        "with_queryid": True,
        "path": bertbase_cat_msmarcopassage_train_scores_ids,
    }
