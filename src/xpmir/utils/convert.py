from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from experimaestro import Config

Input = TypeVar("Input")
Output = TypeVar("Output")


class Converter(Config, ABC, Generic[Input, Output]):
    @abstractmethod
    def __call__(self, input: Input) -> Output:
        pass
