import sys
import types
from xpmir.utils.utils import easylog

_logger = easylog()


# hack for pylint
def FP16_Optimizer(*args, **kwargs):
    return None


def FusedAdam(*args, **kwargs):
    return None


class ApexWrapper(types.ModuleType):
    @property
    def _apex(self):
        try:
            import apex

            return apex
        except ImportError:
            _logger.warn(
                "Module apex not installed. Please see <https://github.com/NVIDIA/apex>"
            )
            raise

    @property
    def FP16_Optimizer(self):
        return self._apex.optimizers.FP16_Optimizer

    @property
    def FusedAdam(self):
        return self._apex.optimizers.FusedAdam


sys.modules[__name__].__class__ = ApexWrapper
