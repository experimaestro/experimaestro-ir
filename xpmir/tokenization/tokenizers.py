import re
from typing import Iterator, NamedTuple, Protocol, Tuple


def basictokenizer(text: str) -> Iterator[Tuple[str, int, int]]:
    for m in re.finditer(r"\w+", text):
        yield m.group(), m.start(), m.end()
