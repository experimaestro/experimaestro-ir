from experimaestro import ObjectStore
import torch.nn as nn
from itertools import chain
from xpmir.learning.optim import Module, ModuleList


class MyModule(Module):
    def __post_init__(self) -> None:
        self.linear = nn.Linear(2, 3)


def test_module_list():
    a = MyModule()
    b = MyModule()
    container = ModuleList(sub_modules=[a, b])

    store = ObjectStore()
    container = container.instance(objects=store)
    a = a.instance(objects=store)
    b = b.instance(objects=store)
    assert set(container.parameters()) == set(chain(a.parameters(), b.parameters()))
