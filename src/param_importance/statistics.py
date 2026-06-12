from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class TestResult:
    statistic: float
    pvalue: float


def one_sided_mean_greater(values: np.ndarray, null: float = 0.0) -> TestResult:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2:
        return TestResult(np.nan, np.nan)
    result = stats.ttest_1samp(clean, popmean=null, alternative="greater")
    return TestResult(float(result.statistic), float(result.pvalue))


def paired_mean_greater(left: np.ndarray, right: np.ndarray) -> TestResult:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    mask = np.isfinite(left_values) & np.isfinite(right_values)
    if mask.sum() < 2:
        return TestResult(np.nan, np.nan)
    result = stats.ttest_rel(left_values[mask], right_values[mask], alternative="greater")
    return TestResult(float(result.statistic), float(result.pvalue))


def tost_one_sample(values: np.ndarray, margin: float) -> dict[str, float | bool]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size < 2 or margin <= 0:
        return {
            "pvalue": np.nan,
            "lower_pvalue": np.nan,
            "upper_pvalue": np.nan,
            "equivalent": False,
        }
    mean = clean.mean()
    standard_error = clean.std(ddof=1) / np.sqrt(clean.size)
    if standard_error == 0:
        equivalent = -margin < mean < margin
        return {
            "pvalue": 0.0 if equivalent else 1.0,
            "lower_pvalue": 0.0 if mean > -margin else 1.0,
            "upper_pvalue": 0.0 if mean < margin else 1.0,
            "equivalent": equivalent,
        }
    degrees = clean.size - 1
    lower_statistic = (mean + margin) / standard_error
    upper_statistic = (mean - margin) / standard_error
    lower_pvalue = 1 - stats.t.cdf(lower_statistic, degrees)
    upper_pvalue = stats.t.cdf(upper_statistic, degrees)
    pvalue = max(lower_pvalue, upper_pvalue)
    return {
        "pvalue": float(pvalue),
        "lower_pvalue": float(lower_pvalue),
        "upper_pvalue": float(upper_pvalue),
        "equivalent": bool(pvalue < 0.05),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    repetitions: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    chunk_size = min(1_000, repetitions)
    completed = 0
    while completed < repetitions:
        current = min(chunk_size, repetitions - completed)
        indices = rng.integers(0, clean.size, size=(current, clean.size))
        means[completed : completed + current] = clean[indices].mean(axis=1)
        completed += current
    alpha = (1 - confidence) / 2
    return (
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def bootstrap_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    repetitions: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    top = np.asarray(numerator, dtype=float)
    bottom = np.asarray(denominator, dtype=float)
    mask = np.isfinite(top) & np.isfinite(bottom)
    top = top[mask]
    bottom = bottom[mask]
    if top.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    ratios = np.empty(repetitions)
    for index in range(repetitions):
        selected = rng.integers(0, top.size, size=top.size)
        ratios[index] = top[selected].mean() / max(bottom[selected].mean(), 1e-30)
    return (
        float(top.mean() / max(bottom.mean(), 1e-30)),
        float(np.quantile(ratios, 0.025)),
        float(np.quantile(ratios, 0.975)),
    )

