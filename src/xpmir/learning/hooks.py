import logging
from experimaestro import Param
from xpmir.learning.context import TrainState, InitializationTrainingHook
from xpmir.learning.parameters import ParametersIterator
from xpmir.utils.utils import easylog

logger = easylog()
logger.setLevel(logging.INFO)


class LayerFreezer(InitializationTrainingHook):
    """This training hook class can be used to freeze some of the transformer layers"""

    selector: Param[ParametersIterator]
    """How to select the layers to freeze"""

    def __init__(self):
        self._initialized = False

    def after(self, state: TrainState):
        if not self._initialized:
            self._initialized = True
            for name, param, to_freeze in self.selector.iter():
                if to_freeze:
                    logger.info("Freezing layer %s", name)
                    param.requires_grad = False
