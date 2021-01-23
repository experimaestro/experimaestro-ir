from typing import ClassVar, Optional, TypeVar, Type, Union

from typing_extensions import Annotated, get_type_hints


class Param:
    def __init__(self, help: str = None):
        self.help = help


class A:
    x: Annotated[int, Param(help="help")] = 2

    def a(self):
        return self.x + 1


print(get_type_hints(A, include_extras=True))


a = A()
x = a.a()
print(x)
