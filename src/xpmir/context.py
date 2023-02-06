from typing import Callable, DefaultDict, Dict, List, Type, TypeVar
from experimaestro import Config
from xpmir.letor import DeviceInformation


class Hook(Config):
    """Base class for all hooks"""

    pass


HookType = TypeVar("HookType")


class InitializationHook(Hook):
    """Base class for hooks before/after initialization"""

    def after(self, context: "Context"):
        """Called after initialization"""
        pass

    def before(self, context: "Context"):
        """Called before initialization"""
        pass


class Context:
    """Generic computational context"""

    hooksmap: Dict[Type, List[Hook]]
    """Map of hooks"""

    def __init__(self, device_information: DeviceInformation, hooks: List[Hook] = []):
        self.device_information = device_information
        self.hooksmap = DefaultDict(lambda: [])
        for hook in hooks:
            self.add_hook(hook)

    def hooks(self, cls: Type[HookType]) -> List[HookType]:
        """Returns all the hooks"""
        return self.hooksmap.get(cls, [])  # type: ignore

    def call_hooks(self, cls: Type, method: Callable, *args, **kwargs):
        for hook in self.hooks(cls):
            method(hook, *args, **kwargs)

    def add_hook(self, hook):
        for cls in hook.__class__.__mro__:
            self.hooksmap[cls].append(hook)
