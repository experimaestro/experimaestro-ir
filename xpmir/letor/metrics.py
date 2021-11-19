from typing import Dict


class MetricAccumulator:
    """Utility class to accumulate a set of metrics over batches
    of (potentially) different sizes"""

    def __init__(self):
        self.metrics = {}
        self.count = 0

    def update(self, metrics: Dict[str, float], count):
        self.count += count
        self.metrics = {
            key: value * count + self.metrics.get(key, 0.0)
            for key, value in metrics.items()
        }

    def merge(self, other: "MetricAccumulator"):
        self.count += other.count
        for key, value in other.metrics.items():
            self.metrics[key] = self.metrics.get(key, 0.0) + value

    def compute(self):
        return {key: value / self.count for key, value in self.metrics.items()}

    def report(self, epoch, writer, prefix):
        metrics = self.compute()
        for key, value in metrics.items():
            writer.add_scalar(
                f"{prefix}/{key}",
                value,
                epoch,
            )
