from __future__ import annotations

import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


@dataclass(slots=True)
class CostMeasurement:
    elapsed_seconds: float = 0.0
    peak_memory_bytes: int = 0


@contextmanager
def measure_cost(device: torch.device | str = "cpu") -> Iterator[CostMeasurement]:
    resolved = torch.device(device)
    result = CostMeasurement()
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
        torch.cuda.reset_peak_memory_stats(resolved)
    start = time.perf_counter()
    try:
        yield result
    finally:
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)
            result.peak_memory_bytes = int(torch.cuda.max_memory_allocated(resolved))
        result.elapsed_seconds = time.perf_counter() - start


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device

