"""Utility functions when aligning two tokenizations"""

from typing import (
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    Iterable,
    Tuple,
    TypeVar,
)
from dataclasses import dataclass


class TokenWithRange(Protocol):
    """A token with its range"""

    string: str
    start: int
    end: int


QueryToken = TypeVar("QueryToken", bound=TokenWithRange)
DocumentToken = TypeVar("DocumentToken", bound=TokenWithRange)


@dataclass
class Match(Generic[QueryToken, DocumentToken]):
    string: str
    query: List[QueryToken]
    document: List[DocumentToken]


def findmatches(
    query_toks: Iterator[QueryToken], document_toks: Iterator[DocumentToken], all=True
) -> Dict[str, Match[QueryToken, DocumentToken]]:
    """Find query tokens in documents"""
    # Find all words in query
    matching = {}

    for token in query_toks:
        if all or token.string not in matching:
            match = matching.setdefault(token.string, Match(token.string, [], []))
            match.query.append(token)

    # Find all query words in documents
    for token in document_toks:
        match = matching.get(token.string, None)
        if match is not None:
            if all or len(match.document) == 0:
                match.document.append(token)

    return matching


def optnext(it: Iterator[TokenWithRange]) -> Optional[TokenWithRange]:
    try:
        return next(it)
    except StopIteration:
        return None


def findpositions(
    keys: Iterator[QueryToken], tokens: Iterator[DocumentToken]
) -> Iterator[Tuple[QueryToken, DocumentToken]]:
    """Find the position of `keys` in `tokens`

    supposes that:

    - keys and tokens are sorted, i.e. start_{i+1} >= end_i
    - all the keys should appear in tokens


    Returns tuples (c, )
    """

    c = optnext(keys)
    if c is None:
        return

    for token in tokens:
        # no match to be done (query)
        if c is None:
            break

        if token.start <= c.start and token.end > c.start:
            yield (c, token)
            c = optnext(keys)

    assert c is None, "Some tokens have not been found"
