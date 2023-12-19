from typing import TypeVar, Iterable
from attrs import define
from xpmir.letor.records import BaseRecords, TopicRecord

RT = TypeVar("RT")


class History:
    pass


@define()
class HistoryRecord(TopicRecord):
    """A topic record with a history context"""

    history: History
