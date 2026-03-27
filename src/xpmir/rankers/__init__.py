# This package contains all rankers
from typing import (
    Generic,
    Protocol,
    TypeVar,
)
from datamaestro_ir.data import Documents
from .retriever import (  # noqa: F401
    Retriever,
    RetrieverHydrator,
    RunRetriever,
    MultiRunRetrieverFactory,
)
from .scorer import (  # noqa: F401
    Scorer,
    RandomScorer,
    TwoStageRetriever,
    scorer_retriever,
    AbstractModuleScorer,
    DuoLearnableScorer,
    DuoTwoStageRetriever,
)

import logging

logger = logging.getLogger(__name__)


ARGS = TypeVar("ARGS")
KWARGS = TypeVar("KWARGS")
T = TypeVar("T")


class DocumentsFunction(Protocol, Generic[KWARGS, ARGS, T]):
    def __call__(self, documents: Documents, *args: ARGS, **kwargs: KWARGS) -> T: ...


def document_cache(fn: DocumentsFunction[KWARGS, ARGS, T]):
    """Decorator

    Allows to cache the result of a function that depends
    on the document dataset ID
    """
    retrievers = {}

    def _fn(*args: ARGS, **kwargs: KWARGS):
        def cached(documents: Documents) -> T:
            dataset_id = documents.__identifier__().all

            if dataset_id not in retrievers:
                retrievers[dataset_id] = fn(documents, *args, **kwargs)

            return retrievers[dataset_id]

        return cached

    return _fn
