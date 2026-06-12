from __future__ import annotations

import copy
import itertools
import math
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import config_hash
from .data import (
    PermutedDataset,
    balanced_subset,
    classification_dataset,
    split_class_tasks,
)
from .estimators import single_direct_from_moments
from .gradients import FunctionalGradientComputer
from .io import git_commit, is_completed, prepare_run_dir, write_metadata, write_parquet
from .models import MLP, TaskAwareResNet18
from .utils import resolve_device, seed_everything


def _forward(model: nn.Module, inputs: torch.Tensor, task_id: int) -> torch.Tensor:
    if isinstance(model, TaskAwareResNet18):
        return model(inputs, task_id)
    return model(inputs)


def _trainable_parameters(model: nn.Module) -> OrderedDict[str, nn.Parameter]:
    return OrderedDict(
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def _clone_parameter_data(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, parameter.detach().clone())
        for name, parameter in _trainable_parameters(model).items()
    )


def _flatten_named(values: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([value.reshape(-1) for value in values.values()])


def _unflatten_like(
    vector: torch.Tensor,
    template: OrderedDict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    cursor = 0
    for name, value in template.items():
        count = value.numel()
        result[name] = vector[cursor : cursor + count].reshape_as(value)
        cursor += count
    if cursor != vector.numel():
        raise ValueError("Vector length does not match parameter template")
    return result


def _regularization_penalty(
    model: nn.Module,
    omega: OrderedDict[str, torch.Tensor] | None,
    anchor: OrderedDict[str, torch.Tensor] | None,
    strength: float,
) -> torch.Tensor:
    parameters = _trainable_parameters(model)
    if omega is None or anchor is None or strength == 0:
        first = next(iter(parameters.values()))
        return first.new_zeros(())
    penalty = next(iter(parameters.values())).new_zeros(())
    for name, parameter in parameters.items():
        penalty = penalty + (omega[name] * (parameter - anchor[name]).square()).sum()
    return 0.5 * strength * penalty


def _make_tasks(
    config: dict[str, Any],
    seed: int,
) -> tuple[list[Dataset], list[Dataset], tuple[int, ...], int]:
    scenario = str(config.get("scenario", "permuted_mnist")).lower()
    data_config = config.get("data", {})
    if scenario == "permuted_mnist":
        dataset_name = str(data_config.get("name", "mnist"))
        train, shape, classes = classification_dataset(
            dataset_name,
            str(data_config.get("root", "data")),
            train=True,
            download=bool(data_config.get("download", False)),
            seed=seed,
            fake_size=int(data_config.get("fake_train_size", 2_000)),
        )
        test, _, _ = classification_dataset(
            dataset_name,
            str(data_config.get("root", "data")),
            train=False,
            download=bool(data_config.get("download", False)),
            seed=seed + 1,
            fake_size=int(data_config.get("fake_test_size", 1_000)),
        )
        train_size = int(data_config.get("train_size", len(train)))
        test_size = int(data_config.get("test_size", len(test)))
        train = balanced_subset(train, min(train_size, len(train)), seed + 31)
        test = balanced_subset(test, min(test_size, len(test)), seed + 37)
        task_count = int(config.get("task_count", 5))
        train_tasks = []
        test_tasks = []
        for task_id in range(task_count):
            generator = torch.Generator().manual_seed(seed * 1009 + task_id)
            permutation = torch.randperm(int(np.prod(shape)), generator=generator)
            train_tasks.append(PermutedDataset(train, permutation))
            test_tasks.append(PermutedDataset(test, permutation))
        return train_tasks, test_tasks, shape, classes

    if scenario == "split_cifar100":
        train, shape, _ = classification_dataset(
            str(data_config.get("name", "cifar100")),
            str(data_config.get("root", "data")),
            train=True,
            download=bool(data_config.get("download", False)),
            seed=seed,
        )
        test, _, _ = classification_dataset(
            str(data_config.get("name", "cifar100")),
            str(data_config.get("root", "data")),
            train=False,
            download=bool(data_config.get("download", False)),
            seed=seed + 1,
        )
        train_size = int(data_config.get("train_size", len(train)))
        test_size = int(data_config.get("test_size", len(test)))
        train = balanced_subset(train, min(train_size, len(train)), seed + 31)
        test = balanced_subset(test, min(test_size, len(test)), seed + 37)
        task_count = int(config.get("task_count", 10))
        classes_per_task = int(config.get("classes_per_task", 10))
        return (
            split_class_tasks(train, classes_per_task, task_count),
            split_class_tasks(test, classes_per_task, task_count),
            shape,
            classes_per_task,
        )
    raise ValueError(f"Unknown continual-learning scenario: {scenario}")


def _make_model(
    scenario: str,
    shape: tuple[int, ...],
    classes: int,
    task_count: int,
    config: dict[str, Any],
) -> nn.Module:
    if scenario == "permuted_mnist":
        hidden_sizes = config.get("model", {}).get("hidden_sizes", [256, 256])
        return MLP(shape, classes, hidden_sizes)
    if scenario == "split_cifar100":
        return TaskAwareResNet18(classes, task_count)
    raise ValueError(scenario)


@torch.no_grad()
def _accuracy(
    model: nn.Module,
    dataset: Dataset,
    task_id: int,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    model.eval()
    correct = 0
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        predictions = _forward(model, inputs, task_id).argmax(dim=1)
        correct += int((predictions == targets).sum().item())
        count += targets.numel()
    return correct / max(count, 1)


def _task_metrics(accuracy_matrix: np.ndarray) -> dict[str, float]:
    learned_tasks = accuracy_matrix.shape[0]
    final = accuracy_matrix[-1, :learned_tasks]
    average_accuracy = float(np.nanmean(final))
    forgetting_values = []
    backward_transfer_values = []
    for task_id in range(learned_tasks - 1):
        best_before_final = np.nanmax(accuracy_matrix[task_id:, task_id])
        forgetting_values.append(best_before_final - final[task_id])
        backward_transfer_values.append(final[task_id] - accuracy_matrix[task_id, task_id])
    return {
        "final_average_accuracy": average_accuracy,
        "average_forgetting": float(np.mean(forgetting_values)) if forgetting_values else 0.0,
        "backward_transfer": (
            float(np.mean(backward_transfer_values)) if backward_transfer_values else 0.0
        ),
    }


def _independent_batch(iterator: Any, loader: DataLoader) -> tuple[Any, torch.Tensor, torch.Tensor]:
    try:
        inputs, targets = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        inputs, targets = next(iterator)
    return iterator, inputs, targets


def _importance_increment(
    method: str,
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    task_id: int,
    learning_rate: float,
    microbatches: int,
    independent_inputs: torch.Tensor | None = None,
    independent_targets: torch.Tensor | None = None,
    oracle_dataset: Dataset | None = None,
    oracle_batch_size: int = 256,
    workers: int = 0,
) -> torch.Tensor:
    parameters = _trainable_parameters(model)
    update_gradient = torch.cat(
        [
            (parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
            for parameter in parameters.values()
        ]
    )
    normalized = method.lower()
    if normalized == "naive":
        return learning_rate * update_gradient.square()

    forward_args: tuple[object, ...] = (task_id,) if isinstance(model, TaskAwareResNet18) else ()
    computer = FunctionalGradientComputer(
        model,
        inputs.device,
        forward_args=forward_args,
    )
    if normalized == "double":
        if independent_inputs is None or independent_targets is None:
            raise ValueError("Double sampling requires an independent minibatch")
        evaluation_gradient = computer.mean_gradient(
            independent_inputs,
            independent_targets,
        )
        return learning_rate * update_gradient * evaluation_gradient
    if normalized == "single_direct":
        per_sample = computer.per_sample_gradient(inputs, targets)
        mean_gradient = per_sample.mean(0)
        centered = per_sample - mean_gradient
        variance = centered.square().sum(0) / (per_sample.shape[0] - 1)
        corrected, _, _ = single_direct_from_moments(
            update_gradient,
            mean_gradient.unsqueeze(0),
            variance.unsqueeze(0),
            per_sample.shape[0],
            learning_rate,
            (1.0,),
        )
        return corrected
    if normalized.startswith("single_micro"):
        group_count = microbatches
        if inputs.shape[0] % group_count:
            raise ValueError("Batch size must be divisible by microbatch count")
        group_size = inputs.shape[0] // group_count
        group_gradients = []
        for group_id in range(group_count):
            start = group_id * group_size
            stop = start + group_size
            group_gradients.append(
                computer.mean_gradient(inputs[start:stop], targets[start:stop])
            )
        group_values = torch.stack(group_gradients)
        mean_gradient = group_values.mean(0)
        centered = group_values - mean_gradient
        variance = centered.square().sum(0) / (group_count - 1)
        corrected, _, _ = single_direct_from_moments(
            update_gradient,
            mean_gradient.unsqueeze(0),
            variance.unsqueeze(0),
            group_count,
            learning_rate,
            (1.0,),
        )
        return corrected
    if normalized == "oracle":
        if oracle_dataset is None:
            raise ValueError("Oracle method requires the current task dataset")
        full_gradient = computer.full_dataset_gradient(
            oracle_dataset,
            oracle_batch_size,
            workers=workers,
        )
        return learning_rate * update_gradient * full_gradient
    raise ValueError(f"Unsupported SI importance method: {method}")


def _fisher_importance(
    model: nn.Module,
    dataset: Dataset,
    task_id: int,
    device: torch.device,
    batch_size: int,
    workers: int,
    gradient_chunk_size: int,
) -> OrderedDict[str, torch.Tensor]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    forward_args: tuple[object, ...] = (task_id,) if isinstance(model, TaskAwareResNet18) else ()
    computer = FunctionalGradientComputer(
        copy.deepcopy(model).eval(),
        device,
        forward_args=forward_args,
    )
    sum_squares = torch.zeros(computer.parameter_count, device=device)
    count = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        for start in range(0, inputs.shape[0], gradient_chunk_size):
            stop = min(start + gradient_chunk_size, inputs.shape[0])
            per_sample = computer.per_sample_gradient(
                inputs[start:stop],
                targets[start:stop],
            )
            sum_squares += per_sample.square().sum(0)
            count += per_sample.shape[0]
    return computer.unflatten(sum_squares / max(count, 1))


def _run_single(
    config: dict[str, Any],
    method: str,
    strength: float,
    seed: int,
    phase: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    scenario = str(config.get("scenario", "permuted_mnist")).lower()
    train_tasks, test_tasks, shape, classes = _make_tasks(config, seed)
    task_count = len(train_tasks)
    model = _make_model(scenario, shape, classes, task_count, config).to(device)
    training = config.get("training", {})
    learning_rate = float(training.get("learning_rate", 0.01))
    epochs = int(training.get("epochs_per_task", 5 if scenario == "permuted_mnist" else 30))
    batch_size = int(training.get("batch_size", 64))
    workers = int(training.get("workers", 0))
    damping = float(training.get("si_damping", 0.1))
    microbatches = int(training.get("microbatches", 4))
    evaluation_batch_size = int(training.get("evaluation_batch_size", 256))
    gradient_chunk_size = int(training.get("gradient_chunk_size", 8))

    optimizer_name = str(training.get("optimizer", "sgd")).lower()
    momentum = float(training.get("momentum", 0.0))
    weight_decay = float(training.get("weight_decay", 0.0))
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    strict_methods = {"naive", "double", "single_direct", "single_micro", "oracle"}
    base_method = method
    if base_method in strict_methods and (
        optimizer_name != "sgd" or momentum != 0 or weight_decay != 0
    ):
        theory_status = "stress_test_non_sgd"
    else:
        theory_status = "strict_sgd"

    omega: OrderedDict[str, torch.Tensor] | None = None
    anchor: OrderedDict[str, torch.Tensor] | None = None
    accuracy_matrix = np.full((task_count, task_count), np.nan)
    task_rows: list[dict[str, Any]] = []
    total_samples_seen = 0
    start_time = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for task_id, task_dataset in enumerate(train_tasks):
        loader_generator = torch.Generator().manual_seed(seed * 1009 + task_id)
        loader = DataLoader(
            task_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            drop_last=base_method.startswith("single_micro"),
            generator=loader_generator,
        )
        independent_iterator = iter(loader)
        task_start = _clone_parameter_data(model)
        path_importance = torch.zeros(
            sum(parameter.numel() for parameter in _trainable_parameters(model).values()),
            device=device,
        )
        samples_seen = 0
        model.train()
        for _ in range(epochs):
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                independent_inputs = None
                independent_targets = None
                if base_method == "double":
                    independent_iterator, independent_inputs, independent_targets = _independent_batch(
                        independent_iterator,
                        loader,
                    )
                    independent_inputs = independent_inputs.to(device)
                    independent_targets = independent_targets.to(device)

                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(_forward(model, inputs, task_id), targets)
                loss = loss + _regularization_penalty(model, omega, anchor, strength)
                loss.backward()
                if base_method not in {"fine_tuning", "ewc"}:
                    increment = _importance_increment(
                        base_method,
                        model,
                        inputs,
                        targets,
                        task_id,
                        learning_rate,
                        microbatches,
                        independent_inputs,
                        independent_targets,
                        task_dataset if base_method == "oracle" else None,
                        evaluation_batch_size,
                        workers,
                    )
                    path_importance += increment.detach()
                optimizer.step()
                samples_seen += targets.numel()
                total_samples_seen += targets.numel()

        current = _clone_parameter_data(model)
        if base_method == "ewc":
            task_omega = _fisher_importance(
                model,
                task_dataset,
                task_id,
                device,
                evaluation_batch_size,
                workers,
                gradient_chunk_size,
            )
        elif base_method == "fine_tuning":
            task_omega = OrderedDict(
                (name, torch.zeros_like(value)) for name, value in current.items()
            )
        else:
            path_dict = _unflatten_like(path_importance, current)
            task_omega = OrderedDict(
                (
                    name,
                    torch.clamp(path_dict[name], min=0)
                    / ((current[name] - task_start[name]).square() + damping),
                )
                for name in current
            )

        if omega is None:
            omega = task_omega
        else:
            omega = OrderedDict(
                (name, omega[name] + task_omega[name]) for name in omega
            )
        anchor = current

        for evaluated_task in range(task_id + 1):
            accuracy_matrix[task_id, evaluated_task] = _accuracy(
                model,
                test_tasks[evaluated_task],
                evaluated_task,
                device,
                evaluation_batch_size,
                workers,
            )
        task_rows.append(
            {
                "phase": phase,
                "method": method,
                "strength": strength,
                "seed": seed,
                "task": task_id,
                "current_average_accuracy": float(
                    np.nanmean(accuracy_matrix[task_id, : task_id + 1])
                ),
                "samples_seen": samples_seen,
                "importance_l1": float(
                    sum(value.abs().sum().item() for value in (omega or {}).values())
                ),
            }
        )

    elapsed = time.perf_counter() - start_time
    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    metrics = _task_metrics(accuracy_matrix)
    summary = {
        "phase": phase,
        "scenario": scenario,
        "method": method,
        "strength": strength,
        "seed": seed,
        **metrics,
        "elapsed_seconds": elapsed,
        "samples_seen": total_samples_seen,
        "peak_memory_bytes": peak_memory,
        "theory_status": theory_status,
        "accuracy_matrix": accuracy_matrix.tolist(),
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
    }
    return summary, task_rows


def _method_base(method: str) -> tuple[str, int | None]:
    if method.startswith("single_micro_m"):
        return "single_micro", int(method.rsplit("m", 1)[1])
    return method, None


def run_continual_experiment(config: dict[str, Any], force: bool = False) -> Path:
    run_dir = prepare_run_dir(config, "continual")
    if is_completed(run_dir) and not force:
        return run_dir
    write_metadata(run_dir, config, "continual", "running")

    methods = [str(value) for value in config.get(
        "methods",
        [
            "fine_tuning",
            "ewc",
            "naive",
            "double",
            "single_direct",
            "single_micro_m4",
            "single_micro_m8",
        ],
    )]
    selection = config.get("selection", {})
    tuning_seeds = [int(value) for value in selection.get("tuning_seeds", [100, 101, 102])]
    final_seeds = [int(value) for value in selection.get("final_seeds", [0, 1, 2, 3, 4])]
    strengths = [
        float(value)
        for value in selection.get(
            "strengths",
            [10.0**power for power in range(-3, 4)],
        )
    ]
    no_regularization_strength = 0.0
    progress_path = run_dir / "progress.parquet"
    task_progress_path = run_dir / "task_progress.parquet"
    summaries = (
        pd.read_parquet(progress_path).to_dict("records")
        if progress_path.exists() and not force
        else []
    )
    task_rows = (
        pd.read_parquet(task_progress_path).to_dict("records")
        if task_progress_path.exists() and not force
        else []
    )
    completed = {
        (row["phase"], row["method"], float(row["strength"]), int(row["seed"]))
        for row in summaries
    }

    def execute(method: str, strength: float, seed: int, phase: str) -> None:
        key = (phase, method, float(strength), seed)
        if key in completed:
            return
        base_method, microbatches = _method_base(method)
        run_config = copy.deepcopy(config)
        if microbatches is not None:
            run_config.setdefault("training", {})["microbatches"] = microbatches
        summary, current_tasks = _run_single(
            run_config,
            base_method,
            strength,
            seed,
            phase,
        )
        summary["method"] = method
        for row in current_tasks:
            row["method"] = method
        summaries.append(summary)
        task_rows.extend(current_tasks)
        completed.add(key)
        write_parquet(pd.DataFrame(summaries), progress_path)
        write_parquet(pd.DataFrame(task_rows), task_progress_path)

    selected_strengths: dict[str, float] = {}
    for method in methods:
        if method == "fine_tuning":
            selected_strengths[method] = no_regularization_strength
            continue
        for strength, seed in itertools.product(strengths, tuning_seeds):
            execute(method, strength, seed, "tuning")
        tuning = pd.DataFrame(
            [
                row
                for row in summaries
                if row["phase"] == "tuning" and row["method"] == method
            ]
        )
        selected = (
            tuning.groupby("strength")["final_average_accuracy"]
            .mean()
            .sort_values(ascending=False)
            .index[0]
        )
        selected_strengths[method] = float(selected)

    for method in methods:
        strength = selected_strengths[method]
        for seed in final_seeds:
            execute(method, strength, seed, "final")

    results = pd.DataFrame(summaries)
    tasks = pd.DataFrame(task_rows)
    write_parquet(results, run_dir / "results.parquet")
    write_parquet(tasks, run_dir / "task_results.parquet")
    selection_frame = pd.DataFrame(
        [
            {"method": method, "selected_strength": strength}
            for method, strength in selected_strengths.items()
        ]
    )
    write_parquet(selection_frame, run_dir / "selected_strengths.parquet")
    write_metadata(
        run_dir,
        config,
        "continual",
        "completed",
        {
            "rows": len(results),
            "task_rows": len(tasks),
            "selected_strengths": selected_strengths,
        },
    )
    return run_dir
