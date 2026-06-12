from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import config_hash
from .io import git_commit, is_completed, prepare_run_dir, write_metadata, write_parquet


@dataclass(frozen=True, slots=True)
class Population:
    name: str
    mean: float
    variance: float
    sampler: Any
    replacement: bool = True
    finite_size: int | None = None


def _normal_population(spec: dict[str, Any]) -> Population:
    mean = float(spec.get("mean", 0.5))
    std = float(spec.get("std", 1.0))

    def sampler(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        return rng.normal(mean, std, size=shape)

    return Population("gaussian", mean, std * std, sampler)


def _student_population(spec: dict[str, Any]) -> Population:
    df = float(spec.get("df", 5.0))
    mean = float(spec.get("mean", 0.5))
    target_std = float(spec.get("std", 1.0))
    if df <= 2:
        raise ValueError("Student-t validation requires df > 2 for finite variance")
    scale = target_std / math.sqrt(df / (df - 2))

    def sampler(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        return mean + scale * rng.standard_t(df, size=shape)

    return Population(f"student_t_{df:g}", mean, target_std**2, sampler)


def _lognormal_population(spec: dict[str, Any]) -> Population:
    log_std = float(spec.get("log_std", 0.8))
    target_mean = float(spec.get("mean", 0.5))
    raw_mean = math.exp(log_std * log_std / 2)
    raw_variance = (math.exp(log_std * log_std) - 1) * math.exp(log_std * log_std)

    def sampler(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        centered = rng.lognormal(0.0, log_std, size=shape) - raw_mean
        return centered + target_mean

    return Population("centered_lognormal", target_mean, raw_variance, sampler)


def _quadratic_population(spec: dict[str, Any]) -> Population:
    theta = float(spec.get("theta", 1.0))
    data_mean = float(spec.get("data_mean", 0.25))
    data_std = float(spec.get("data_std", 1.0))
    gradient_mean = theta - data_mean

    def sampler(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        z = rng.normal(data_mean, data_std, size=shape)
        return theta - z

    return Population("quadratic_gradient", gradient_mean, data_std**2, sampler)


def _linear_population(spec: dict[str, Any]) -> Population:
    theta = float(spec.get("theta", 1.0))
    beta = float(spec.get("beta", 0.5))
    noise_std = float(spec.get("noise_std", 0.5))
    coefficient = theta - beta
    gradient_mean = coefficient
    gradient_variance = 2 * coefficient**2 + noise_std**2

    def sampler(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        x = rng.normal(size=shape)
        epsilon = rng.normal(scale=noise_std, size=shape)
        return coefficient * x * x - x * epsilon

    return Population("linear_regression_gradient", gradient_mean, gradient_variance, sampler)


def _logistic_population(spec: dict[str, Any]) -> Population:
    theta = float(spec.get("theta", 1.0))
    beta = float(spec.get("beta", 0.25))
    sigmoid = lambda value: 1.0 / (1.0 + math.exp(-value))
    gradient_mean = sigmoid(theta) - sigmoid(beta)
    label_variance = sigmoid(beta) * (1 - sigmoid(beta))

    def sampler(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        x = rng.choice(np.array([-1.0, 1.0]), size=shape)
        probability = 1.0 / (1.0 + np.exp(-beta * x))
        y = rng.binomial(1, probability)
        return (1.0 / (1.0 + np.exp(-theta * x)) - y) * x

    return Population("logistic_regression_gradient", gradient_mean, label_variance, sampler)


def _finite_population(spec: dict[str, Any], seed: int) -> Population:
    size = int(spec.get("population_size", 10_000))
    mean = float(spec.get("mean", 0.5))
    std = float(spec.get("std", 1.0))
    population_rng = np.random.default_rng(seed)
    values = population_rng.normal(mean, std, size=size)
    actual_mean = float(values.mean())
    actual_variance = float(values.var(ddof=0))

    def sampler(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
        repetitions, batch_size = shape
        if batch_size > size:
            raise ValueError("Batch size exceeds finite population size")
        result = np.empty(shape, dtype=np.float64)
        for row in range(repetitions):
            result[row] = rng.choice(values, size=batch_size, replace=False)
        return result

    return Population(
        "finite_population_without_replacement",
        actual_mean,
        actual_variance,
        sampler,
        replacement=False,
        finite_size=size,
    )


def make_population(spec: dict[str, Any], seed: int) -> Population:
    name = str(spec["name"]).lower()
    factories = {
        "gaussian": _normal_population,
        "student_t": _student_population,
        "centered_lognormal": _lognormal_population,
        "quadratic": _quadratic_population,
        "linear_regression": _linear_population,
        "logistic_regression": _logistic_population,
    }
    if name == "finite_population":
        return _finite_population(spec, seed)
    try:
        return factories[name](spec)
    except KeyError as error:
        raise ValueError(f"Unknown simulation population/problem: {name}") from error


def theoretical_gaussian_variances(
    mean: float,
    variance: float,
    batch_size: int,
    microbatches: int,
    gamma: float,
) -> dict[str, float]:
    sigma4 = variance * variance
    common = 4 * mean * mean * variance / batch_size
    return {
        "single_direct": gamma**2
        * (common + 2 * sigma4 / (batch_size * (batch_size - 1))),
        "single_micro": gamma**2
        * (common + 2 * microbatches * sigma4 / (batch_size**2 * (microbatches - 1))),
        "double": gamma**2 * (common + 4 * sigma4 / batch_size**2),
        "double_matched": gamma**2
        * (common + 2 * microbatches * sigma4 / batch_size**2),
    }


def _simulate_chunk(
    population: Population,
    rng: np.random.Generator,
    repetitions: int,
    batch_size: int,
    microbatches: int,
    gamma: float,
) -> pd.DataFrame:
    if batch_size % microbatches:
        raise ValueError(f"batch_size={batch_size} must be divisible by M={microbatches}")
    if batch_size % 2:
        raise ValueError("Double sampling requires an even total batch size")

    samples = population.sampler(rng, (repetitions, batch_size)).astype(np.float64)
    sample_mean = samples.mean(axis=1)
    sample_variance = samples.var(axis=1, ddof=1)
    naive = gamma * sample_mean**2
    direct = gamma * (sample_mean**2 - sample_variance / batch_size)

    groups = samples.reshape(repetitions, microbatches, batch_size // microbatches)
    group_means = groups.mean(axis=2)
    micro = gamma * (
        sample_mean**2 - group_means.var(axis=1, ddof=1) / microbatches
    )
    double_matched = gamma * np.mean(
        group_means[:, 0::2] * group_means[:, 1::2],
        axis=1,
    )

    half = batch_size // 2
    if population.replacement:
        mean_a = samples[:, :half].mean(axis=1)
        mean_b = samples[:, half:].mean(axis=1)
    else:
        # Independent finite-population draws match the double-sampling protocol.
        samples_a = population.sampler(rng, (repetitions, half))
        samples_b = population.sampler(rng, (repetitions, half))
        mean_a = samples_a.mean(axis=1)
        mean_b = samples_b.mean(axis=1)
    double = gamma * mean_a * mean_b
    oracle = gamma * population.mean**2

    return pd.DataFrame(
        {
            "oracle": oracle,
            "naive": naive,
            "double": double,
            "double_matched": double_matched,
            "single_direct": direct,
            "single_micro": micro,
            "sample_variance": sample_variance,
        }
    )


def run_simulation(config: dict[str, Any], force: bool = False) -> Path:
    run_dir = prepare_run_dir(config, "simulate")
    if is_completed(run_dir) and not force:
        return run_dir

    write_metadata(run_dir, config, "simulate", "running")
    repetitions = int(config.get("repetitions", 20_000))
    chunk_size = int(config.get("chunk_size", min(2_000, repetitions)))
    gamma = float(config.get("gamma", 0.01))
    base_seed = int(config.get("seed", 0))
    batch_sizes = [int(value) for value in config.get("batch_sizes", [8, 32, 128])]
    microbatch_counts = [int(value) for value in config.get("microbatches", [2, 4, 8, 16])]
    population_specs = config.get(
        "populations",
        [
            {"name": "gaussian"},
            {"name": "student_t", "df": 5},
            {"name": "centered_lognormal"},
            {"name": "finite_population"},
        ],
    )

    frames: list[pd.DataFrame] = []
    grid_index = 0
    for population_spec in population_specs:
        population = make_population(population_spec, base_seed + grid_index)
        for batch_size in batch_sizes:
            valid_microbatches = [m for m in microbatch_counts if m <= batch_size and batch_size % m == 0]
            for microbatches in valid_microbatches:
                rng = np.random.default_rng(base_seed + grid_index * 10_003)
                completed = 0
                while completed < repetitions:
                    current = min(chunk_size, repetitions - completed)
                    frame = _simulate_chunk(
                        population,
                        rng,
                        current,
                        batch_size,
                        microbatches,
                        gamma,
                    )
                    frame["repetition"] = np.arange(completed, completed + current)
                    frame["population"] = population.name
                    frame["population_mean"] = population.mean
                    frame["population_variance"] = population.variance
                    frame["replacement"] = population.replacement
                    frame["iid_theory_applicable"] = population.replacement
                    frame["batch_size"] = batch_size
                    frame["microbatches"] = microbatches
                    frame["gamma"] = gamma
                    frame["seed"] = base_seed
                    frame["sample_count"] = batch_size
                    frame["config_hash"] = config_hash(config)
                    frame["git_commit"] = git_commit()
                    theory_applies = population.name in {
                        "gaussian",
                        "quadratic_gradient",
                    }
                    theory = (
                        theoretical_gaussian_variances(
                            population.mean,
                            population.variance,
                            batch_size,
                            microbatches,
                            gamma,
                        )
                        if theory_applies
                        else {
                            "single_direct": math.nan,
                            "single_micro": math.nan,
                            "double": math.nan,
                        }
                    )
                    frame["gaussian_variance_formula_applicable"] = theory_applies
                    for key, value in theory.items():
                        frame[f"theory_variance_{key}"] = value
                    frames.append(frame)
                    completed += current
                grid_index += 1

    result = pd.concat(frames, ignore_index=True)
    write_parquet(result, run_dir / "results.parquet")
    summary_rows = []
    for keys, group in result.groupby(["population", "batch_size", "microbatches"]):
        for estimator in [
            "naive",
            "double",
            "double_matched",
            "single_direct",
            "single_micro",
        ]:
            errors = group[estimator] - group["oracle"]
            summary_rows.append(
                {
                    "population": keys[0],
                    "batch_size": keys[1],
                    "microbatches": keys[2],
                    "estimator": estimator,
                    "mean_error": float(errors.mean()),
                    "variance": float(group[estimator].var(ddof=1)),
                    "mse": float(np.mean(errors**2)),
                    "expected_naive_bias_iid": (
                        gamma * float(group["population_variance"].iloc[0]) / keys[1]
                        if estimator == "naive"
                        and bool(group["iid_theory_applicable"].iloc[0])
                        else math.nan
                    ),
                }
            )
    write_parquet(pd.DataFrame(summary_rows), run_dir / "summary.parquet")
    write_metadata(
        run_dir,
        config,
        "simulate",
        "completed",
        {"rows": len(result), "summary_rows": len(summary_rows)},
    )
    return run_dir
