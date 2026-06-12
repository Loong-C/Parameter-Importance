from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import config_hash
from .data import balanced_subset, batch_from_indices, classification_dataset
from .estimators import (
    double_estimate,
    microbatch_estimate,
    oracle_estimate,
    ppt_variance_only_ablation,
    single_direct_from_moments,
)
from .gradients import FunctionalGradientComputer, group_slices, select_slices
from .io import git_commit, is_completed, prepare_run_dir, write_metadata, write_parquet
from .models import build_model
from .quadrature import QuadratureRule, get_rule
from .utils import measure_cost, resolve_device, seed_everything


def _optimizer(model: nn.Module, spec: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(spec.get("name", "sgd")).lower()
    learning_rate = float(spec.get("learning_rate", 0.01))
    weight_decay = float(spec.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(spec.get("momentum", 0.0)),
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {name}")


def _checkpoint_epochs(total_epochs: int, fractions: list[float]) -> list[int]:
    return sorted(
        {
            min(total_epochs, max(0, int(round(total_epochs * fraction))))
            for fraction in fractions
        }
    )


def _gradient_group_counts(
    batch_size: int,
    requested_counts: list[int],
) -> tuple[list[int], list[int]]:
    if batch_size % 2:
        raise ValueError("Double sampling requires even checkpoint batch sizes")
    valid_requested = [
        count
        for count in requested_counts
        if count <= batch_size and batch_size % count == 0
    ]
    computed = sorted(set([2, *valid_requested]))
    return valid_requested, computed


def _train_and_save_checkpoints(
    model: nn.Module,
    dataset: Dataset,
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> dict[int, Path]:
    training = config.get("training", {})
    total_epochs = int(training.get("epochs", 4))
    fractions = [float(value) for value in training.get("checkpoint_fractions", [0, 0.25, 0.5, 1])]
    epochs = _checkpoint_epochs(total_epochs, fractions)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}

    initial_path = checkpoint_dir / "epoch-000.pt"
    torch.save(model.state_dict(), initial_path)
    paths[0] = initial_path
    if total_epochs == 0:
        return paths

    loader_generator = torch.Generator().manual_seed(int(config.get("seed", 0)))
    loader = DataLoader(
        dataset,
        batch_size=int(training.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(training.get("workers", 0)),
        generator=loader_generator,
    )
    optimizer = _optimizer(model, training.get("optimizer", {}))
    model.train()
    for epoch in range(1, total_epochs + 1):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), targets)
            loss.backward()
            optimizer.step()
        if epoch in epochs:
            path = checkpoint_dir / f"epoch-{epoch:03d}.pt"
            torch.save(model.state_dict(), path)
            paths[epoch] = path
    return paths


def _node_parameter_sets(
    computer: FunctionalGradientComputer,
    delta: torch.Tensor,
    rules: list[QuadratureRule],
) -> dict[float, dict[str, torch.Tensor]]:
    nodes = sorted({float(node) for rule in rules for node in rule.nodes})
    return {node: computer.shifted_params(delta, node) for node in nodes}


def _integrated(values: torch.Tensor, rule: QuadratureRule) -> torch.Tensor:
    weights = torch.as_tensor(rule.weights, dtype=values.dtype, device=values.device)
    return torch.einsum("q,qd->d", weights, values)


def _matched_double_average(
    group_u: torch.Tensor,
    group_v: torch.Tensor,
    gamma: float,
    weights: tuple[float, ...],
) -> torch.Tensor:
    if group_u.shape[0] % 2:
        raise ValueError("Matched double sampling requires an even group count")
    estimates = []
    for first in range(0, group_u.shape[0], 2):
        second = first + 1
        estimates.append(
            double_estimate(
                group_u[first : first + 1],
                group_v[:, first : first + 1],
                group_u[second : second + 1],
                group_v[:, second : second + 1],
                gamma,
                weights,
                symmetric=True,
            )
        )
    return torch.stack(estimates).mean(0)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or np.all(left == left[0]) or np.all(right == right[0]):
        return math.nan
    return float(spearmanr(left, right).statistic)


def _metric_rows(
    *,
    estimator: str,
    raw: torch.Tensor,
    correction: torch.Tensor,
    estimate: torch.Tensor,
    oracle: torch.Tensor,
    quadrature_oracle: torch.Tensor,
    layout_groups: dict[str, list[slice]],
    metadata: dict[str, Any],
    loss_decrease: float,
    backward_count: int,
    elapsed_seconds: float,
    peak_memory_bytes: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_name, slices in layout_groups.items():
        selected_raw = select_slices(raw, slices).detach().cpu()
        selected_correction = select_slices(correction, slices).detach().cpu()
        selected_estimate = select_slices(estimate, slices).detach().cpu()
        selected_oracle = select_slices(oracle, slices).detach().cpu()
        selected_quadrature = select_slices(quadrature_oracle, slices).detach().cpu()
        error = selected_estimate - selected_oracle
        oracle_scale = float(selected_oracle.abs().sum().item())
        rows.append(
            {
                **metadata,
                "estimator": estimator,
                "aggregation": "model" if group_name == "all" else "layer",
                "group": group_name,
                "raw_value": float(selected_raw.mean().item()),
                "correction": float(selected_correction.mean().item()),
                "corrected_value": float(selected_estimate.mean().item()),
                "oracle": float(selected_oracle.mean().item()),
                "signed_bias": float(error.mean().item()),
                "absolute_bias": float(error.abs().mean().item()),
                "relative_bias": float(error.sum().item() / (oracle_scale + 1e-12)),
                "mse": float(error.square().mean().item()),
                "spearman": _safe_spearman(
                    selected_estimate.numpy(),
                    selected_oracle.numpy(),
                ),
                "sum_estimate": float(selected_estimate.sum().item()),
                "sum_oracle": float(selected_oracle.sum().item()),
                "quadrature_error": float(
                    (selected_quadrature - selected_oracle).abs().mean().item()
                ),
                "conservation_error": float(selected_estimate.sum().item() - loss_decrease),
                "loss_decrease": loss_decrease,
                "backward_count": backward_count,
                "elapsed_seconds": elapsed_seconds,
                "peak_memory_bytes": peak_memory_bytes,
            }
        )
    return rows


def _parameter_rows(
    estimate: torch.Tensor,
    oracle: torch.Tensor,
    estimator: str,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    estimate_cpu = estimate.detach().cpu().numpy()
    oracle_cpu = oracle.detach().cpu().numpy()
    return [
        {
            **metadata,
            "estimator": estimator,
            "parameter_index": index,
            "estimate": float(value),
            "oracle": float(oracle_cpu[index]),
            "error": float(value - oracle_cpu[index]),
        }
        for index, value in enumerate(estimate_cpu)
    ]


def run_checkpoint_experiment(config: dict[str, Any], force: bool = False) -> Path:
    run_dir = prepare_run_dir(config, "checkpoint")
    required = ("results.parquet", "checkpoints")
    if is_completed(run_dir, required) and not force:
        return run_dir
    write_metadata(run_dir, config, "checkpoint", "running")

    seed = int(config.get("seed", 0))
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    data_config = config.get("data", {})
    dataset, input_shape, num_classes = classification_dataset(
        str(data_config.get("name", "fake")),
        str(data_config.get("root", "data")),
        train=True,
        download=bool(data_config.get("download", False)),
        seed=seed,
        fake_size=int(data_config.get("fake_size", 2_000)),
        augmentation=bool(data_config.get("augmentation", False)),
    )
    population = balanced_subset(
        dataset,
        int(data_config.get("population_size", min(10_000, len(dataset)))),
        seed + 17,
    )
    model_config = config.get("model", {})
    model_name = str(model_config.get("name", "mlp"))
    model = build_model(
        model_name,
        input_shape,
        num_classes,
        **{key: value for key, value in model_config.items() if key != "name"},
    ).to(device)
    checkpoint_paths = _train_and_save_checkpoints(model, dataset, config, run_dir, device)

    experiment = config.get("experiment", {})
    gamma = float(experiment.get("gamma", 0.01))
    batch_sizes = [int(value) for value in experiment.get("batch_sizes", [16, 64, 256])]
    microbatch_counts = [int(value) for value in experiment.get("microbatches", [2, 4, 8])]
    repetitions = int(experiment.get("repetitions", 512))
    gradient_chunk_size = int(experiment.get("gradient_chunk_size", 16))
    oracle_batch_size = int(experiment.get("oracle_batch_size", 256))
    replacement = bool(experiment.get("replacement", True))
    rules = [get_rule(str(value)) for value in experiment.get("quadrature_rules", ["single"])]
    reference_rule = get_rule(str(experiment.get("oracle_rule", rules[-1].name)))
    all_rules = rules + ([reference_rule] if reference_rule.name not in {rule.name for rule in rules} else [])
    store_parameter_metrics = bool(experiment.get("store_parameter_metrics", False))
    optimizer_spec = config.get("training", {}).get("optimizer", {})
    strict_sgd = (
        str(optimizer_spec.get("name", "sgd")).lower() == "sgd"
        and float(optimizer_spec.get("momentum", 0.0)) == 0
        and float(optimizer_spec.get("weight_decay", 0.0)) == 0
    )

    rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    population_rng = np.random.default_rng(seed + 101)
    commit = git_commit()
    current_hash = config_hash(config)

    for checkpoint_epoch, checkpoint_path in sorted(checkpoint_paths.items()):
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        computer = FunctionalGradientComputer(copy.deepcopy(model), device)
        full_update_gradient = computer.full_dataset_gradient(
            population,
            oracle_batch_size,
        )
        delta = -gamma * full_update_gradient
        parameter_sets = _node_parameter_sets(computer, delta, all_rules)
        full_node_gradients = {
            node: computer.full_dataset_gradient(
                population,
                oracle_batch_size,
                params,
            )
            for node, params in parameter_sets.items()
        }
        shifted_end = computer.shifted_params(delta, 1.0)
        loss_before = computer.full_dataset_loss(population, oracle_batch_size)
        loss_after = computer.full_dataset_loss(
            population,
            oracle_batch_size,
            shifted_end,
        )
        loss_decrease = loss_before - loss_after
        layout_groups = group_slices(computer.layout)

        reference_values = torch.stack(
            [full_node_gradients[float(node)] for node in reference_rule.nodes]
        )
        reference_oracle = oracle_estimate(
            full_update_gradient,
            reference_values,
            gamma,
            reference_rule.weights,
        )

        for rule in rules:
            node_params = [parameter_sets[float(node)] for node in rule.nodes]
            full_values = torch.stack(
                [full_node_gradients[float(node)] for node in rule.nodes]
            )
            quadrature_oracle = oracle_estimate(
                full_update_gradient,
                full_values,
                gamma,
                rule.weights,
            )
            for batch_size in batch_sizes:
                if batch_size > len(population) and not replacement:
                    continue
                valid_microbatches, computed_group_counts = _gradient_group_counts(
                    batch_size,
                    microbatch_counts,
                )
                for repetition in range(repetitions):
                    indices = population_rng.choice(
                        len(population),
                        size=batch_size,
                        replace=replacement,
                    )
                    inputs, targets = batch_from_indices(population, indices, device)
                    with measure_cost(device) as cost:
                        moments, grouped_gradients = (
                            computer.paired_cross_moments_with_groups(
                                inputs,
                                targets,
                                [computer.params, *node_params],
                                gradient_chunk_size,
                                computed_group_counts,
                            )
                        )
                        assert moments.mean_x is not None
                        assert moments.mean_y is not None
                        mean_u = moments.mean_x
                        mean_v = moments.mean_y[1:]
                        variance_u = moments.sample_cross_covariance[0]
                        cross_covariance = moments.sample_cross_covariance[1:]
                        raw = oracle_estimate(mean_u, mean_v, gamma, rule.weights)
                        direct, _, direct_correction = single_direct_from_moments(
                            mean_u,
                            mean_v,
                            cross_covariance,
                            batch_size,
                            gamma,
                            rule.weights,
                        )
                        ppt, _, ppt_correction = ppt_variance_only_ablation(
                            mean_u,
                            mean_v,
                            variance_u,
                            batch_size,
                            gamma,
                            rule.weights,
                        )

                        double_all_u, double_all_v = grouped_gradients[2]
                        double_u = double_all_u
                        double_v = double_all_v[1:]
                        double = double_estimate(
                            double_u[:1],
                            double_v[:, :1],
                            double_u[1:],
                            double_v[:, 1:],
                            gamma,
                            rule.weights,
                            symmetric=True,
                        )

                        micro_results: dict[
                            int,
                            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                        ] = {}
                        for microbatches in valid_microbatches:
                            all_group_u, all_group_v = grouped_gradients[microbatches]
                            group_u = all_group_u
                            group_v = all_group_v[1:]
                            micro, _, micro_correction = microbatch_estimate(
                                group_u,
                                group_v,
                                gamma,
                                rule.weights,
                            )
                            matched_double = _matched_double_average(
                                group_u,
                                group_v,
                                gamma,
                                rule.weights,
                            )
                            micro_results[microbatches] = (
                                micro,
                                micro_correction,
                                matched_double,
                            )

                    common = {
                        "config_hash": current_hash,
                        "git_commit": commit,
                        "seed": seed,
                        "repetition": repetition,
                        "dataset": str(data_config.get("name", "fake")),
                        "model": model_name,
                        "checkpoint": f"epoch-{checkpoint_epoch:03d}",
                        "batch_size": batch_size,
                        "sample_count": batch_size,
                        "quadrature": rule.name,
                        "oracle_quadrature": reference_rule.name,
                        "replacement": replacement,
                        "optimizer": str(
                            optimizer_spec.get("name", "sgd")
                        ),
                        "theory_status": (
                            "strict_sgd_iid_sampling"
                            if strict_sgd and replacement
                            else "strict_sgd_finite_population_sampling"
                            if strict_sgd
                            else "checkpoint_distribution_only_non_sgd"
                        ),
                    }
                    estimator_values = {
                        "naive": (raw, torch.zeros_like(raw), raw, len(rule.nodes) + 1),
                        "double": (
                            double,
                            raw - double,
                            double,
                            2 * (len(rule.nodes) + 1),
                        ),
                        "single_direct": (
                            raw,
                            direct_correction,
                            direct,
                            batch_size * (len(rule.nodes) + 1),
                        ),
                        "ppt_variance_only": (
                            raw,
                            ppt_correction,
                            ppt,
                            batch_size * (len(rule.nodes) + 1),
                        ),
                    }
                    for estimator, (estimator_raw, correction, estimate, backward_count) in estimator_values.items():
                        rows.extend(
                            _metric_rows(
                                estimator=estimator,
                                raw=estimator_raw,
                                correction=correction,
                                estimate=estimate,
                                oracle=reference_oracle,
                                quadrature_oracle=quadrature_oracle,
                                layout_groups=layout_groups,
                                metadata={**common, "microbatches": math.nan},
                                loss_decrease=loss_decrease,
                                backward_count=backward_count,
                                elapsed_seconds=cost.elapsed_seconds,
                                peak_memory_bytes=cost.peak_memory_bytes,
                            )
                        )
                        if store_parameter_metrics:
                            parameter_rows.extend(
                                _parameter_rows(
                                    estimate,
                                    reference_oracle,
                                    estimator,
                                    {**common, "microbatches": math.nan},
                                )
                            )
                    for microbatches, (
                        estimate,
                        correction,
                        matched_double,
                    ) in micro_results.items():
                        estimator_name = f"single_micro_m{microbatches}"
                        rows.extend(
                            _metric_rows(
                                estimator=estimator_name,
                                raw=raw,
                                correction=correction,
                                estimate=estimate,
                                oracle=reference_oracle,
                                quadrature_oracle=quadrature_oracle,
                                layout_groups=layout_groups,
                                metadata={**common, "microbatches": microbatches},
                                loss_decrease=loss_decrease,
                                backward_count=microbatches * (len(rule.nodes) + 1),
                                elapsed_seconds=cost.elapsed_seconds,
                                peak_memory_bytes=cost.peak_memory_bytes,
                            )
                        )
                        if store_parameter_metrics:
                            parameter_rows.extend(
                                _parameter_rows(
                                    estimate,
                                    reference_oracle,
                                    estimator_name,
                                    {**common, "microbatches": microbatches},
                                )
                            )
                        rows.extend(
                            _metric_rows(
                                estimator=f"double_matched_m{microbatches}",
                                raw=matched_double,
                                correction=torch.zeros_like(matched_double),
                                estimate=matched_double,
                                oracle=reference_oracle,
                                quadrature_oracle=quadrature_oracle,
                                layout_groups=layout_groups,
                                metadata={**common, "microbatches": microbatches},
                                loss_decrease=loss_decrease,
                                backward_count=microbatches * (len(rule.nodes) + 1),
                                elapsed_seconds=cost.elapsed_seconds,
                                peak_memory_bytes=cost.peak_memory_bytes,
                            )
                        )

    result = pd.DataFrame(rows)
    write_parquet(result, run_dir / "results.parquet")
    if parameter_rows:
        write_parquet(pd.DataFrame(parameter_rows), run_dir / "parameter_results.parquet")
    write_metadata(
        run_dir,
        config,
        "checkpoint",
        "completed",
        {
            "rows": len(result),
            "parameter_rows": len(parameter_rows),
            "device": str(device),
        },
    )
    return run_dir
