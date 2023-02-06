from typing import Optional, List
from experimaestro import Param
from xpmir.utils.utils import foreach
from . import Retriever


class SumRetriever(Retriever):
    """Combines the scores of various retrievers"""

    retrievers: Param[List[Retriever]]
    """retrievers"""

    weights: Param[Optional[List[int]]] = None
    """The weights of the retrievers"""

    def __validate__(self):
        super().__validate__()
        assert self.weights is None or len(self.weights) == len(self.retrievers)

    def initialize(self):
        super().initialize()
        foreach(self.retrievers, lambda retriever: retriever.initialize())
