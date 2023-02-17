from attrs import define
from xpmir.papers.monobert.configuration import Monobert, Learner


@define(kw_only=True)
class DuoBERT(Monobert):
    duobert: Learner
