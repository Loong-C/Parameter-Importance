from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import quad_vec


@dataclass(frozen=True, slots=True)
class QuadratureRule:
    name: str
    nodes: tuple[float, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.nodes) != len(self.weights):
            raise ValueError("Quadrature nodes and weights must have equal length")
        if not np.isclose(sum(self.weights), 1.0):
            raise ValueError("Quadrature weights on [0, 1] must sum to one")


@dataclass(frozen=True, slots=True)
class AdaptiveQuadratureResult:
    value: np.ndarray
    estimated_error: float
    success: bool
    status: int
    evaluations: int
    intervals: np.ndarray
    interval_errors: np.ndarray


def get_rule(name: str, points: int | None = None) -> QuadratureRule:
    normalized = name.lower().replace("-", "_")
    if normalized in {"single", "endpoint", "si"}:
        return QuadratureRule("single", (0.0,), (1.0,))
    if normalized in {"midpoint", "mid"}:
        return QuadratureRule("midpoint", (0.5,), (1.0,))
    if normalized in {"simpson", "simpson3", "simpson_3"}:
        return QuadratureRule("simpson3", (0.0, 0.5, 1.0), (1 / 6, 4 / 6, 1 / 6))
    if normalized in {"gauss3", "gauss_legendre3", "gauss_legendre_3"}:
        points = 3
    elif normalized in {"gauss16", "gauss_legendre16", "gauss_legendre_16"}:
        points = 16
    elif normalized in {"gauss", "gauss_legendre"}:
        points = points or 3
    else:
        raise ValueError(f"Unknown quadrature rule: {name}")

    nodes, weights = np.polynomial.legendre.leggauss(points)
    mapped_nodes = (nodes + 1.0) / 2.0
    mapped_weights = weights / 2.0
    return QuadratureRule(
        f"gauss_legendre_{points}",
        tuple(float(value) for value in mapped_nodes),
        tuple(float(value) for value in mapped_weights),
    )


def adaptive_vector_integral(
    function: Callable[[float], np.ndarray],
    *,
    epsabs: float = 1e-8,
    epsrel: float = 1e-5,
    norm: str = "2",
    limit: int = 256,
    quadrature: str = "gk21",
    points: list[float] | None = None,
    cache_size: int = 100_000_000,
) -> AdaptiveQuadratureResult:
    value, error, info = quad_vec(
        function,
        0.0,
        1.0,
        epsabs=epsabs,
        epsrel=epsrel,
        norm=norm,
        limit=limit,
        quadrature=quadrature,
        points=points,
        cache_size=cache_size,
        full_output=True,
    )
    return AdaptiveQuadratureResult(
        value=np.asarray(value, dtype=np.float64),
        estimated_error=float(error),
        success=bool(info.success),
        status=int(info.status),
        evaluations=int(info.neval),
        intervals=np.asarray(info.intervals, dtype=np.float64),
        interval_errors=np.asarray(info.errors, dtype=np.float64),
    )


def composite_trapezoid_vector(
    function: Callable[[float], np.ndarray],
    intervals: int,
) -> np.ndarray:
    if intervals < 1:
        raise ValueError("intervals must be positive")
    nodes = np.linspace(0.0, 1.0, intervals + 1)
    values = np.stack([np.asarray(function(float(node))) for node in nodes])
    weights = np.ones(intervals + 1, dtype=np.float64)
    weights[[0, -1]] = 0.5
    return np.tensordot(weights / intervals, values, axes=(0, 0))
