from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class EstimatorResult:
    estimator: str
    raw_value: float
    correction: float
    corrected_value: float
    oracle: float
    sample_count: int
    backward_count: int
    elapsed_seconds: float
    peak_memory_bytes: int
    seed: int
    repetition: int
    config_hash: str
    git_commit: str
    dataset: str = ""
    model: str = ""
    checkpoint: str = ""
    aggregation: str = "model"
    group: str = "all"
    notes: str = ""

    @property
    def error(self) -> float:
        return self.corrected_value - self.oracle

    @property
    def squared_error(self) -> float:
        return self.error * self.error

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["error"] = self.error
        row["squared_error"] = self.squared_error
        return row

