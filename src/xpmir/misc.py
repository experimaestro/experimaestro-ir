from abc import ABC, abstractmethod
from functools import cached_property
from typing import List
from pathlib import Path
from experimaestro import Config, Meta


class IDList(Config, ABC):
    """A configuration that returns a list of ids"""

    @property
    @abstractmethod
    def ids(self) -> List[str]:
        """Returns the list of IDs"""
        ...


class FileIDList(IDList):
    """A file-based list of IDs"""

    path: Meta[Path]

    @cached_property
    def ids(self) -> List[str]:
        return list(self.path.read_text().split("\n"))
