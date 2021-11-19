import torch
from typing import Iterator, List, Tuple

from xpmir.letor.records import BaseRecords


RecordsGenerator = Iterator[Tuple[BaseRecords, List[float]]]


class MemoryStatistics:
    def reset(self):
        pass

    def max_allocated(self) -> int:
        raise NotImplementedError()

    def max(self) -> int:
        raise NotImplementedError()


class CudaMemoryStatistics:
    def reset(self):
        torch.cuda.reset_accumulated_memory_stats()

    def max_allocated(self):
        return torch.cuda.max_memory_allocated()

    def total(self) -> int:
        return torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).total_memory


def generate_texts(maxlen: int):
    for textlen in np.random.choice(maxlen, count, replace=False):
        yield (textlen, self.ranker.generate_text(l))


class MemoryPredictor:
    def __init__(self):
        self.predictors = None

    def predictor(self):
        """Returns a predictor"""
        assert self.predictors is not None
        return self.predictors[1] if torch.is_grad_enabled() else self.predictors[0]

    @property
    def calibrated(self):
        return self.predictors is not None

    def calibrate(self, generator: RecordsGenerator, memory: MemoryStatistics):
        """Tries to calibrate a model

        Arguments:

            count: How many samples should be taken
        """
        from sklearn import linear_model

        device = torch.cuda.current_device()

        self.ranker.zero_grad(set_to_none=True)
        usedbytes = torch.cuda.memory_allocated(device)

        # LASSO optimization with
        # query q/document d token length (a ql + b ql²)(c dl + dl²) + constant = used bytes
        # both with and without grad

        data = [[], []]
        targets = [[], []]
        for records, statistics in generator:
            for withgrad in [0, 1]:
                with torch.set_grad_enabled(withgrad == 1):
                    self.ranker(records)
                    memory.reset()
                    deltabytes = memory.max_allocated() - usedbytes
                    assert deltabytes > 0

                    data[withgrad].append(statistics)
                    targets[withgrad].append(deltabytes)

        self.predictors = []
        for withgrad in [0, 1]:
            clf = linear_model.Lasso(alpha=1)
            clf.fit(data[withgrad], targets)
            self.predictors.append(clf)
