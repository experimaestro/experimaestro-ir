from pathlib import Path
from typing import Iterator, Optional
import torch
import torch.nn as nn
from experimaestro import Param
from xpmir.learning.optim import Module, ModuleLoader
from xpmir.learning.parameters import (
    ParameterElement,
    ParametersIterator,
    SubParametersIterator,
    InverseParametersIterator,
    SubModuleLoader,
)


class Model(Module, nn.Module):
    sub_model: Param[Optional["Model"]]

    def __post_init__(self):
        self.layer = nn.Linear(1, 1)


class MyParametersIterator(ParametersIterator):
    model: Param[Model]

    def iter(self) -> Iterator[ParameterElement]:
        for name, param in self.model.named_parameters():
            yield ParameterElement(name, param, param is self.model.layer.weight)


def test_iterator():
    model = Model(sub_model=Model())
    pi = MyParametersIterator(model=model)

    for x in pi.instance().iter():
        assert x.selected == (x.name == "layer.weight")


def test_inverse_iterator():
    model = Model(sub_model=Model())
    pi = InverseParametersIterator(iterator=MyParametersIterator(model=model))

    for x in pi.instance().iter():
        assert x.selected == (x.name != "layer.weight")


def test_sub_iterator():
    sub = Model()
    model = Model(sub_model=sub)
    sub_iterator = MyParametersIterator(model=sub)
    pi = SubParametersIterator(
        model=model, default=False, iterator=sub_iterator
    ).instance()

    for x in pi.iter():
        assert x.selected == (x.name == "sub_model.layer.weight")


def test_SubModuleLoader(tmp_path: Path):
    """Test the partial module loader"""
    sub = Model()
    model = Model(sub_model=sub)
    datapath = tmp_path / "data.pth"
    loader = ModuleLoader(path=datapath, value=model)
    loader.__xpm__.task = loader  # For testing

    # Use an instance and save the parameters
    model_instance = model.instance()
    with datapath.open("wb") as fp:
        torch.save(model_instance.state_dict(), fp)

    # creates an instance with loader
    sub = Model()
    model = Model(sub_model=sub)
    sub_instance = sub.instance()
    assert model_instance.sub_model.layer.weight != sub_instance.layer.weight

    # creates an instance without loader
    sub = Model()
    model = Model(sub_model=sub)
    sub_selector = MyParametersIterator(model=sub)
    sub_loader = SubModuleLoader.from_module_loader(loader, model, sub, sub_selector)
    sub.add_pretasks(sub_loader)
    sub_instance = sub.instance()
    assert model_instance.sub_model.layer.weight == sub_instance.layer.weight
