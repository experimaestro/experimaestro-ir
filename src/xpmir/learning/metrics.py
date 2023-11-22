from typing import Dict
import logging
from torch.utils.tensorboard.writer import SummaryWriter


class Metric:
    """Represents a generic metric"""

    key: str
    """The key for this metric (used for logging)"""

    count: int
    """Number of samples for this value"""

    def __init__(self, key, count):
        self.key = key
        self.count = count

    def merge(self, other: "Metric"):
        assert other.__class__ is self.__class__
        self._merge(other)

    def _merge(self, other: "Metric"):
        raise NotImplementedError(f"_merge in {self.__class__}")

    def report(self, epoch: int, writer: SummaryWriter, prefix: str):
        raise NotImplementedError(f"report in {self.__class__}")


class ScalarMetric(Metric):
    """Represents a scalar metric"""

    sum: float

    def __init__(self, key: str, value: float, count: int):
        super().__init__(key, count)
        self.sum = value * count

    def _merge(self, other: "ScalarMetric"):
        self.sum += other.sum
        self.count += other.count

    def report(self, step: int, writer: SummaryWriter, prefix: str):
        if self.count == 0:
            logging.warning("Count is 0 when reporting metrics")
        writer.add_scalar(
            f"{prefix}/{self.key}",
            self.sum / self.count,
            step,
        )


class Metrics:
    """Utility class to accumulate a set of metrics over batches
    of (potentially) different sizes"""

    def __init__(self):
        self.metrics: Dict[str, Metric] = {}

    def add(self, metric: Metric):
        if metric.key in self.metrics:
            self.metrics[metric.key].merge(metric)
        else:
            self.metrics[metric.key] = metric

    def merge(self, other: "Metrics"):
        for value in other.metrics.values():
            self.add(value)

    def report(self, epoch, writer, prefix):
        for value in self.metrics.values():
            value.report(epoch, writer, prefix)
