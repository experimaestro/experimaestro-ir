from typing import Optional
from abc import ABC, abstractmethod
import torch.nn as nn
from typing import Iterator, NamedTuple
from experimaestro import Param, Config, PathSerializationLWTask
import torch
from xpmir.learning.optim import Module, ModuleLoader
import logging

logger = logging.getLogger("xpmir.learning")


class ParameterElement(NamedTuple):
    name: str
    """Name of the parameter (with respect to the main model)

    See
    https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_parameters
    """

    parameter: nn.parameter.Parameter
    """The parameter object"""

    selected: bool
    """Selection status"""

    def __repr__(self):
        return (
            f"(name={self.name}, id(parameter)={id(self.parameter)},"
            f" selected={self.selected})"
        )


class ParametersIterator(Config, ABC):
    """Iterator over module parameters

    This can be useful to freeze some layers, or perform any other
    parameter-wise operation
    """

    @abstractmethod
    def iter(self) -> Iterator[ParameterElement]:
        """Iterate over parameters

        :yield: An iterator over triplets (name of the parameter, parameters,
            selected or not)
        """
        ...


class InverseParametersIterator(ParametersIterator):
    """Inverse the selection of a parameter iterator"""

    iterator: Param[ParametersIterator]

    def iter(self) -> Iterator[ParameterElement]:
        yield from (
            ParameterElement(name, param, not (selected))
            for name, param, selected in self.iterator.iter()
        )


class SubParametersIterator(ParametersIterator):
    """Wraps a parameter iterator over a global model and a selector
    over a subpart of the model"""

    model: Param[Module]
    """The model from which the parameters should be gathered"""

    iterator: Param[ParametersIterator]
    """The sub-model iterator"""

    default: Param[bool]
    """Default value for parameters not within the sub-model"""

    def iter(self) -> Iterator[ParameterElement]:
        # Gather all the model parameters
        model_params = {
            id(p): ParameterElement(name, p, self.default)
            for name, p in self.model.named_parameters()
        }

        # Copy the selection status from the sub-model iterator
        for name, param, selected in self.iterator.iter():
            if mp := model_params.get(id(param), None):
                if mp.selected != selected:
                    model_params[id(param)] = ParameterElement(
                        mp.name, mp.parameter, selected
                    )
            else:
                raise RuntimeError("Sub-model parameters are not model parameters")

        # Return everything
        yield from model_params.values()


class NameMapper(Config, ABC):
    """Changes name of parameters"""

    @abstractmethod
    def __call__(self, source: str) -> str:
        ...


class PrefixRenamer(NameMapper):
    """Changes name of parameters"""

    model: Param[str]
    """Prefix in model"""

    data: Param[str]
    """Prefix in data"""

    def __call__(self, source: str) -> str:
        if source.startswith(self.model):
            return f"{self.data}{source[len(self.model):]}"

        return source


class PartialModuleLoader(PathSerializationLWTask):
    """Allows to load only a part of the parameters"""

    selector: Param[ParametersIterator]
    """The selectors gives the list of parameters for which some"""

    mapper: Param[Optional[NameMapper]] = None
    """Maps parameter names so it matches so the saved ones"""

    def execute(self):
        """Combine the model in the selectors"""
        self.value.initialize(None)
        data = torch.load(self.path)
        logger.info(
            "(partial module loader) Loading parameters from %s into %s",
            self.path,
            type(self.value).__name__,
        )

        partial_data = {}
        value_names = set(key for key, _ in self.value.named_parameters())
        for name, _, selected in self.selector.iter():
            if selected:
                if self.mapper is None:
                    key = name
                    logger.debug(f"Selected: {name}")
                else:
                    key = self.mapper(name)
                    logger.debug(f"Selected: {key} -> {name}")

                assert key in data, (
                    f"{key} is not in loaded parameters:" f"{', '.join(data.keys())}"
                )
                partial_data[name] = data[key]

        # Log some potentially useful information
        data_names = set(partial_data.keys())
        inter_names = value_names.intersection(data_names)
        not_used = data_names.difference(inter_names)

        if len(not_used) > 0:
            logger.error("Some selected parameters are not model parameters")
            logger.error("Unused parameters: %s", ", ".join(not_used))

            logger.error("Model parameters: %s", ", ".join(value_names))
            logger.error("Data parameters: %s", ", ".join(data.keys()))
            raise RuntimeError("Some selected parameters are not model parameters")

        assert len(inter_names) > 0, "No common parameters?"

        # Loads with strict False since some keys might not be
        # in the data
        self.value.load_state_dict(partial_data, strict=False)

    @staticmethod
    def from_module_loader(
        module_loader: "ModuleLoader",
        value: Config,
        selector: ParametersIterator,
        *,
        mapper: Optional[NameMapper] = None,
    ):
        """Creates a new partial module loader from another module loader

        :param module_loader: The module loader
        :param value: The module for which parameters should be loaded
        :param selector: The selector to restrict the set of parameters
        :return: A new partial module loader
        """
        assert (
            module_loader.__xpm__.task is not None
        ), "No task associated with the module_loader"

        pml = PartialModuleLoader(
            selector=selector, value=value, path=module_loader.path, mapper=mapper
        )
        pml.copy_dependencies(module_loader)
        return pml


class SubModuleLoader(PathSerializationLWTask):
    """Allows to load only a part of the parameters (with automatic renaming)"""

    selector: Param[ParametersIterator]
    """The selectors gives the list of parameters for which
    loaded parameters should be used"""

    saved_value: Param[Optional[Module]] = None
    """The original module that is being loaded (optional,
    allows to map names)"""

    def execute(self):
        """Combine the model in the selectors"""
        self.value.initialize(None)
        data = torch.load(self.path)
        logger.info(
            "(partial module loader) Loading parameters from %s into %s",
            self.path,
            type(self.value).__name__,
        )

        # Creates a mapper if needed
        mapper = None
        if self.saved_value:
            mapper = {
                id(params): key for key, params in self.saved_value.named_parameters()
            }

        partial_data = {}
        value_names = set(key for key, _ in self.value.named_parameters())
        for name, params, selected in self.selector.iter():
            if selected:
                if mapper is None:
                    key = name
                    logger.debug(f"Selected: {name}")
                else:
                    key = mapper[id(params)]
                    logger.debug(f"Selected: {key} -> {name}")

                assert key in data, (
                    f"{key} is not in loaded parameters:" f"{', '.join(data.keys())}"
                )
                partial_data[name] = data[key]

        # Log some potentially useful information
        data_names = set(partial_data.keys())
        inter_names = value_names.intersection(data_names)
        not_used = data_names.difference(inter_names)

        if len(not_used) > 0:
            logger.error("Some selected parameters are not model parameters")
            logger.error("Unused parameters: %s", ", ".join(not_used))

            logger.error("Model parameters: %s", ", ".join(value_names))
            logger.error("Data parameters: %s", ", ".join(data.keys()))
            raise RuntimeError("Some selected parameters are not model parameters")

        assert len(inter_names) > 0, "No common parameters?"

        # Loads with strict False since some keys might not be
        # in the data
        self.value.load_state_dict(partial_data, strict=False)

    @staticmethod
    def from_module_loader(
        module_loader: "ModuleLoader",
        saved_value: Optional[Config],
        value: Config,
        selector: ParametersIterator,
    ):
        """Creates a new partial module loader from another module loader

        :param module_loader: The module loader
        :param saved_value: The configuration which has been saved
        :param value: The module for which parameters should be loaded (should
            be a sub-module of saved_value). If None, uses the module_loader
            value
        :param selector: The selector to restrict the set of parameters
        :return: A new partial module loader
        """
        assert (
            module_loader.__xpm__.task is not None
        ), "No task associated with the module_loader"

        assert isinstance(module_loader, ModuleLoader)

        pml = SubModuleLoader(
            selector=selector,
            value=value,
            path=module_loader.path,
            saved_value=module_loader.value if saved_value is None else saved_value,
        )
        pml.copy_dependencies(module_loader)
        return pml
