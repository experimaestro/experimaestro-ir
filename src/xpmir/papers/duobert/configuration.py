from attrs import define, Factory
from xpmir.papers.monobert.configuration import (
    Monobert,
    Learner as BaseLearner,
    Retrieval as BaseRetrieval,
)


@define(kw_only=True)
class Retrieval(BaseRetrieval):
    base_k: int = 30


@define(kw_only=True)
class Learner(BaseLearner):
    base_validation_top_k: int = 30


@define(kw_only=True)
class DuoBERT(Monobert):
    duobert: Learner = Factory(Learner)
    retrieval: Retrieval = Factory(Retrieval)
