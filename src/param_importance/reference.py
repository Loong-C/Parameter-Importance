from __future__ import annotations

import copy
import glob
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import config_hash
from .data import balanced_subset, classification_dataset
from .gradients import FunctionalGradientComputer, group_slices, select_slices
from .io import git_commit, is_completed, prepare_run_dir, write_metadata, write_parquet
from .models import build_model
from .quadrature import adaptive_vector_integral, composite_trapezoid_vector
from .utils import resolve_device, seed_everything


def _source_run_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(value) for value in glob.glob(pattern)]
        if not matches and Path(pattern).exists():
            matches = [Path(pattern)]
        paths.extend(path for path in matches if path.is_dir())
    result = sorted(set(path.resolve() for path in paths))
    if not result:
        raise FileNotFoundError("No checkpoint source runs matched reference.source_runs")
    return result


def _load_source_config(source_run: Path) -> dict[str, Any]:
    metadata_path = source_run / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "completed":
        raise ValueError(f"Source run is not completed: {source_run}")
    return dict(metadata["config"])


def _checkpoint_paths(source_run: Path, selected: list[str]) -> list[Path]:
    available = sorted((source_run / "checkpoints").glob("epoch-*.pt"))
    if not selected or selected == ["all"]:
        return available
    wanted = {
        value if value.startswith("epoch-") else f"epoch-{int(value):03d}"
        for value in selected
    }
    result = [path for path in available if path.stem in wanted]
    missing = wanted - {path.stem for path in result}
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints in {source_run}: {sorted(missing)}"
        )
    return result


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.linalg.norm(left - right)
        / max(np.linalg.norm(right), np.finfo(np.float64).tiny)
    )


def _aggregate_rows(
    *,
    source_run: Path,
    checkpoint: str,
    method: str,
    importance: np.ndarray,
    adaptive_importance: np.ndarray,
    layout_groups: dict[str, list[slice]],
    loss_decrease: float,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    tensor = torch.from_numpy(importance)
    adaptive_tensor = torch.from_numpy(adaptive_importance)
    rows: list[dict[str, Any]] = []
    for group_name, slices in layout_groups.items():
        selected = select_slices(tensor, slices).numpy()
        selected_adaptive = select_slices(adaptive_tensor, slices).numpy()
        rows.append(
            {
                **metadata,
                "source_run": source_run.name,
                "checkpoint": checkpoint,
                "method": method,
                "aggregation": "model" if group_name == "all" else "layer",
                "group": group_name,
                "parameter_count": int(selected.size),
                "importance_sum": float(selected.sum()),
                "importance_l1": float(np.abs(selected).sum()),
                "importance_l2": float(np.linalg.norm(selected)),
                "adaptive_difference_l2": float(
                    np.linalg.norm(selected - selected_adaptive)
                ),
                "adaptive_relative_l2": _relative_l2(
                    selected,
                    selected_adaptive,
                ),
                "loss_decrease": loss_decrease,
                "conservation_error": float(selected.sum() - loss_decrease)
                if group_name == "all"
                else np.nan,
                "conservation_relative": float(
                    abs(selected.sum() - loss_decrease)
                    / max(abs(loss_decrease), np.finfo(np.float64).tiny)
                )
                if group_name == "all"
                else np.nan,
            }
        )
    return rows


def run_reference_experiment(config: dict[str, Any], force: bool = False) -> Path:
    run_dir = prepare_run_dir(config, "reference")
    required = ("results.parquet", "diagnostics.parquet", "vectors")
    if is_completed(run_dir, required) and not force:
        return run_dir
    write_metadata(run_dir, config, "reference", "running")

    reference = config.get("reference", {})
    source_runs = _source_run_paths(
        [str(value) for value in reference.get("source_runs", [])]
    )
    selected_checkpoints = [
        str(value) for value in reference.get("checkpoints", ["all"])
    ]
    epsabs = float(reference.get("epsabs", 1e-8))
    epsrel = float(reference.get("epsrel", 1e-4))
    norm = str(reference.get("norm", "2"))
    limit = int(reference.get("limit", 256))
    quadrature = str(reference.get("quadrature", "gk21"))
    points = reference.get("points")
    points = [float(value) for value in points] if points else None
    cache_size = int(reference.get("cache_size", 100_000_000))
    evaluation_cache_entries = int(reference.get("evaluation_cache_entries", 96))
    composite_intervals = [
        int(value) for value in reference.get("composite_intervals", [8, 16, 32])
    ]
    crosscheck_tolerance = float(reference.get("crosscheck_tolerance", 1e-3))
    conservation_tolerance = float(reference.get("conservation_tolerance", 1e-3))
    device = resolve_device(str(config.get("device", "auto")))

    rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    vector_dir = run_dir / "vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    current_hash = config_hash(config)
    commit = git_commit()

    for source_run in source_runs:
        source_config = _load_source_config(source_run)
        seed = int(source_config.get("seed", 0))
        seed_everything(seed)
        data_config = source_config.get("data", {})
        dataset, input_shape, num_classes = classification_dataset(
            str(data_config.get("name", "fake")),
            str(data_config.get("root", "data")),
            train=True,
            download=bool(data_config.get("download", False)),
            seed=seed,
            fake_size=int(data_config.get("fake_size", 2_000)),
            augmentation=False,
        )
        population = balanced_subset(
            dataset,
            int(data_config.get("population_size", min(10_000, len(dataset)))),
            seed + 17,
        )
        model_config = source_config.get("model", {})
        model_name = str(model_config.get("name", "mlp"))
        model = build_model(
            model_name,
            input_shape,
            num_classes,
            **{key: value for key, value in model_config.items() if key != "name"},
        ).to(device)
        experiment = source_config.get("experiment", {})
        gamma = float(experiment.get("gamma", 0.01))
        oracle_batch_size = int(experiment.get("oracle_batch_size", 512))

        for checkpoint_path in _checkpoint_paths(source_run, selected_checkpoints):
            started = time.perf_counter()
            state = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            computer = FunctionalGradientComputer(copy.deepcopy(model), device)
            full_update_gradient = computer.full_dataset_gradient(
                population,
                oracle_batch_size,
            )
            delta = -gamma * full_update_gradient
            delta_numpy = delta.detach().cpu().numpy().astype(np.float64)
            shifted_end = computer.shifted_params(delta, 1.0)
            loss_before = computer.full_dataset_loss(population, oracle_batch_size)
            loss_after = computer.full_dataset_loss(
                population,
                oracle_batch_size,
                shifted_end,
            )
            loss_decrease = loss_before - loss_after
            evaluation_cache: OrderedDict[float, np.ndarray] = OrderedDict()
            evaluation_count = 0

            def full_gradient_at(alpha: float) -> np.ndarray:
                nonlocal evaluation_count
                key = round(float(alpha), 15)
                if key in evaluation_cache:
                    evaluation_cache.move_to_end(key)
                    return evaluation_cache[key]
                params = computer.shifted_params(delta, key)
                gradient = computer.full_dataset_gradient(
                    population,
                    oracle_batch_size,
                    params,
                )
                value = gradient.detach().cpu().numpy().astype(np.float64)
                evaluation_cache[key] = value
                evaluation_count += 1
                if len(evaluation_cache) > evaluation_cache_entries:
                    evaluation_cache.popitem(last=False)
                return value

            adaptive = adaptive_vector_integral(
                full_gradient_at,
                epsabs=epsabs,
                epsrel=epsrel,
                norm=norm,
                limit=limit,
                quadrature=quadrature,
                points=points,
                cache_size=cache_size,
            )
            adaptive_importance = -delta_numpy * adaptive.value
            composite_values: dict[int, np.ndarray] = {}
            for intervals in composite_intervals:
                composite_values[intervals] = composite_trapezoid_vector(
                    full_gradient_at,
                    intervals,
                )
            final_intervals = composite_intervals[-1]
            final_importance = -delta_numpy * composite_values[final_intervals]
            previous_intervals = composite_intervals[-2]
            previous_importance = -delta_numpy * composite_values[previous_intervals]
            crosscheck_relative = _relative_l2(
                final_importance,
                adaptive_importance,
            )
            refinement_relative = _relative_l2(
                final_importance,
                previous_importance,
            )
            conservation_relative = float(
                abs(adaptive_importance.sum() - loss_decrease)
                / max(abs(loss_decrease), np.finfo(np.float64).tiny)
            )
            composite_conservation_relative = float(
                abs(final_importance.sum() - loss_decrease)
                / max(abs(loss_decrease), np.finfo(np.float64).tiny)
            )
            certified = bool(
                adaptive.success
                and crosscheck_relative <= crosscheck_tolerance
                and refinement_relative <= crosscheck_tolerance
                and conservation_relative <= conservation_tolerance
                and composite_conservation_relative <= conservation_tolerance
            )
            common = {
                "config_hash": current_hash,
                "git_commit": commit,
                "seed": seed,
                "dataset": str(data_config.get("name", "fake")),
                "model": model_name,
                "gamma": gamma,
                "population_size": len(population),
                "adaptive_quadrature": quadrature,
                "adaptive_epsabs": epsabs,
                "adaptive_epsrel": epsrel,
                "adaptive_norm": norm,
                "adaptive_success": adaptive.success,
                "adaptive_status": adaptive.status,
                "adaptive_estimated_error": adaptive.estimated_error,
                "adaptive_evaluations": adaptive.evaluations,
                "unique_gradient_evaluations": evaluation_count,
                "crosscheck_intervals": final_intervals,
                "crosscheck_relative_l2": crosscheck_relative,
                "refinement_relative_l2": refinement_relative,
                "conservation_relative_adaptive": conservation_relative,
                "conservation_relative_composite": composite_conservation_relative,
                "reference_certified": certified,
                "elapsed_seconds": time.perf_counter() - started,
            }
            layout_groups = group_slices(computer.layout)
            rows.extend(
                _aggregate_rows(
                    source_run=source_run,
                    checkpoint=checkpoint_path.stem,
                    method=f"adaptive_{quadrature}",
                    importance=adaptive_importance,
                    adaptive_importance=adaptive_importance,
                    layout_groups=layout_groups,
                    loss_decrease=loss_decrease,
                    metadata=common,
                )
            )
            for intervals, integral in composite_values.items():
                importance = -delta_numpy * integral
                rows.extend(
                    _aggregate_rows(
                        source_run=source_run,
                        checkpoint=checkpoint_path.stem,
                        method=f"composite_trapezoid_{intervals}",
                        importance=importance,
                        adaptive_importance=adaptive_importance,
                        layout_groups=layout_groups,
                        loss_decrease=loss_decrease,
                        metadata=common,
                    )
                )
                diagnostic_rows.append(
                    {
                        **common,
                        "source_run": source_run.name,
                        "checkpoint": checkpoint_path.stem,
                        "method": f"composite_trapezoid_{intervals}",
                        "intervals": intervals,
                        "relative_l2_to_adaptive": _relative_l2(
                            importance,
                            adaptive_importance,
                        ),
                        "conservation_relative": float(
                            abs(importance.sum() - loss_decrease)
                            / max(abs(loss_decrease), np.finfo(np.float64).tiny)
                        ),
                    }
                )
            safe_name = f"{source_run.name}-{checkpoint_path.stem}"
            np.savez_compressed(
                vector_dir / f"{safe_name}.npz",
                delta=delta_numpy,
                adaptive_integral=adaptive.value,
                adaptive_importance=adaptive_importance,
                composite_integral=composite_values[final_intervals],
                composite_importance=final_importance,
                adaptive_intervals=adaptive.intervals,
                adaptive_interval_errors=adaptive.interval_errors,
            )

    result = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    write_parquet(result, run_dir / "results.parquet")
    write_parquet(diagnostics, run_dir / "diagnostics.parquet")
    write_metadata(
        run_dir,
        config,
        "reference",
        "completed",
        {
            "rows": len(result),
            "diagnostic_rows": len(diagnostics),
            "sources": [str(path) for path in source_runs],
            "device": str(device),
            "certified_checkpoints": int(
                result.loc[
                    result["aggregation"].eq("model")
                    & result["method"].str.startswith("adaptive_"),
                    "reference_certified",
                ].sum()
            ),
        },
    )
    return run_dir
