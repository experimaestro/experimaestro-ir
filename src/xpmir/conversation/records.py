from typing import TypeVar, Iterable
from xpmir.letor.records import BaseRecords

RT = TypeVar("RT")


class HistoryRecord:
    pass


class BaseConversationRecords(BaseRecords[RT]):
    """Triplets (query, document, history)

    The base class does not impose anything on the structure of the data
    """

    history: Iterable[HistoryRecord]
