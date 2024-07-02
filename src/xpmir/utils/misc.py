from typing import Optional, Sequence, TypeVar, Union


T = TypeVar("T")


def opt_slice(x: Optional[Sequence[T]], ix: Union[int, slice]) -> Optional[Sequence[T]]:
    if x is None:
        return None
    return x[ix]
