from attrs import Factory
from xpmir.papers.helpers import configuration
from xpmir.papers.monobert.configuration import (
    Monobert,
    Learner as BaseLearner,
    Retrieval as BaseRetrieval,
)


@configuration()
class Retrieval(BaseRetrieval):
    base_k: int = 30


@configuration()
class Learner(BaseLearner):
    base_validation_top_k: int = 30


@configuration()
class DuoBERT(Monobert):
    duobert: Learner = Factory(Learner)
    retrieval: Retrieval = Factory(Retrieval)
