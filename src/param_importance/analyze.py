from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from .io import discover_results, prepare_run_dir, write_metadata, write_parquet
from .statistics import (
    bootstrap_mean_ci,
    bootstrap_ratio_ci,
    one_sided_mean_greater,
    paired_mean_greater,
    tost_one_sample,
)


ESTIMATORS = [
    "naive",
    "double",
    "double_matched",
    "single_direct",
    "single_micro",
]


def _simulation_analysis(
    frame: pd.DataFrame,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    group_columns = ["population", "batch_size", "microbatches"]
    for keys, group in frame.groupby(group_columns):
        oracle = group["oracle"].to_numpy()
        errors = {
            estimator: group[estimator].to_numpy() - oracle for estimator in ESTIMATORS
        }
        h1 = one_sided_mean_greater(errors["naive"])
        lower, upper = bootstrap_mean_ci(
            errors["naive"],
            bootstrap_repetitions,
            seed=seed,
        )
        practical_scale = max(float(np.mean(np.abs(oracle))), 1e-12)
        h1_pass = bool(
            h1.pvalue < 0.05
            and lower > 0
            and float(np.mean(errors["naive"])) > 0.1 * practical_scale
        )
        decisions.append(
            {
                "source": "simulate",
                "hypothesis": "H1",
                "population": keys[0],
                "batch_size": keys[1],
                "microbatches": keys[2],
                "estimate": float(np.mean(errors["naive"])),
                "ci_lower": lower,
                "ci_upper": upper,
                "raw_pvalue": h1.pvalue,
                "pass": h1_pass,
                "applicable": True,
                "reason": "positive bias, CI, and 10% practical threshold",
            }
        )

        for estimator in ["double", "single_direct", "single_micro"]:
            margin = 0.05 * practical_scale
            equivalence = tost_one_sample(errors[estimator], margin)
            improvement = paired_mean_greater(
                np.abs(errors["naive"]),
                np.abs(errors[estimator]),
            )
            decisions.append(
                {
                    "source": "simulate",
                    "hypothesis": "H2",
                    "population": keys[0],
                    "batch_size": keys[1],
                    "microbatches": keys[2],
                    "estimator": estimator,
                    "estimate": float(np.mean(errors[estimator])),
                    "equivalence_margin": margin,
                    "raw_pvalue": float(equivalence["pvalue"]),
                    "improvement_pvalue": improvement.pvalue,
                    "pass": bool(
                        equivalence["pvalue"] < 0.05 and improvement.pvalue < 0.05
                    ),
                    "applicable": bool(group["iid_theory_applicable"].iloc[0]),
                    "reason": "TOST equivalence and paired absolute-error improvement",
                }
            )

        direct_squared = errors["single_direct"] ** 2
        micro_squared = errors["single_micro"] ** 2
        double_squared = errors["double_matched"] ** 2
        candidate = micro_squared if keys[2] >= 2 else direct_squared
        ratio, ratio_lower, ratio_upper = bootstrap_ratio_ci(
            candidate,
            double_squared,
            bootstrap_repetitions,
            seed,
        )
        iid_applicable = bool(group["iid_theory_applicable"].iloc[0])
        if keys[2] == 2 and iid_applicable:
            h3_pass = bool(ratio_lower >= 0.95 and ratio_upper <= 1.05)
            reason = "M=2 equivalence within a 5% MSE margin"
        elif iid_applicable:
            h3_pass = bool(ratio_upper < 1.0)
            reason = "single-micro has lower paired MSE at M>2"
        else:
            h3_pass = False
            reason = "outside preregistered i.i.d. equivalence claim"
        decisions.append(
            {
                "source": "simulate",
                "hypothesis": "H3",
                "population": keys[0],
                "batch_size": keys[1],
                "microbatches": keys[2],
                "estimate": ratio,
                "ci_lower": ratio_lower,
                "ci_upper": ratio_upper,
                "raw_pvalue": np.nan,
                "pass": h3_pass,
                "applicable": iid_applicable,
                "reason": reason,
            }
        )

        for estimator in ESTIMATORS:
            values = group[estimator].to_numpy()
            error = errors[estimator]
            groups.append(
                {
                    "source": "simulate",
                    "population": keys[0],
                    "batch_size": keys[1],
                    "microbatches": keys[2],
                    "estimator": estimator,
                    "mean_error": float(error.mean()),
                    "absolute_bias": float(np.abs(error).mean()),
                    "variance": float(values.var(ddof=1)),
                    "mse": float(np.mean(error**2)),
                }
            )
    return decisions, groups


def _checkpoint_analysis(
    frame: pd.DataFrame,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    model_rows = frame[frame["aggregation"] == "model"].copy()
    group_columns = [
        "dataset",
        "model",
        "checkpoint",
        "batch_size",
        "quadrature",
        "estimator",
        "microbatches",
    ]
    summaries = (
        model_rows.groupby(group_columns, dropna=False)
        .agg(
            mean_signed_bias=("signed_bias", "mean"),
            mean_absolute_bias=("absolute_bias", "mean"),
            mse=("mse", "mean"),
            mean_spearman=("spearman", "mean"),
            conservation_error=("conservation_error", "mean"),
            elapsed_seconds=("elapsed_seconds", "mean"),
            peak_memory_bytes=("peak_memory_bytes", "max"),
        )
        .reset_index()
        .assign(source="checkpoint")
        .to_dict("records")
    )
    decisions: list[dict[str, Any]] = []
    condition_columns = ["dataset", "model", "checkpoint", "batch_size", "quadrature"]
    for keys, condition in model_rows.groupby(condition_columns):
        by_estimator = {
            name: values.sort_values(["seed", "repetition"])
            for name, values in condition.groupby("estimator")
        }
        if "naive" not in by_estimator:
            continue
        naive = by_estimator["naive"]
        naive_error = naive["signed_bias"].to_numpy()
        practical_scale = max(float(naive["oracle"].abs().mean()), 1e-12)
        h1 = one_sided_mean_greater(naive_error)
        lower, upper = bootstrap_mean_ci(
            naive_error,
            bootstrap_repetitions,
            seed=seed,
        )
        common = {
            "source": "checkpoint",
            "dataset": keys[0],
            "model": keys[1],
            "checkpoint": keys[2],
            "batch_size": keys[3],
            "quadrature": keys[4],
            "applicable": True,
        }
        decisions.append(
            {
                **common,
                "hypothesis": "H1",
                "estimate": float(np.mean(naive_error)),
                "ci_lower": lower,
                "ci_upper": upper,
                "raw_pvalue": h1.pvalue,
                "pass": bool(
                    h1.pvalue < 0.05
                    and lower > 0
                    and float(np.mean(naive_error)) > 0.1 * practical_scale
                ),
                "reason": "checkpoint positive bias, CI, and 10% practical threshold",
            }
        )

        comparison_names = [
            name
            for name in by_estimator
            if name in {"double", "single_direct"} or name.startswith("single_micro_m")
        ]
        for estimator in comparison_names:
            candidate = by_estimator[estimator]
            merged = naive[
                ["seed", "repetition", "absolute_bias", "mse"]
            ].merge(
                candidate[
                    ["seed", "repetition", "signed_bias", "absolute_bias", "mse"]
                ],
                on=["seed", "repetition"],
                suffixes=("_naive", "_candidate"),
            )
            margin = 0.05 * practical_scale
            equivalence = tost_one_sample(merged["signed_bias"].to_numpy(), margin)
            improvement = paired_mean_greater(
                merged["absolute_bias_naive"].to_numpy(),
                merged["absolute_bias_candidate"].to_numpy(),
            )
            decisions.append(
                {
                    **common,
                    "hypothesis": "H2",
                    "estimator": estimator,
                    "estimate": float(merged["signed_bias"].mean()),
                    "equivalence_margin": margin,
                    "raw_pvalue": float(equivalence["pvalue"]),
                    "improvement_pvalue": improvement.pvalue,
                    "pass": bool(
                        equivalence["pvalue"] < 0.05 and improvement.pvalue < 0.05
                    ),
                    "reason": "checkpoint TOST and paired absolute-error improvement",
                }
            )

        if "double" in by_estimator:
            double = by_estimator["double"][
                ["seed", "repetition", "mse"]
            ].rename(columns={"mse": "mse_double"})
            for estimator, candidate in by_estimator.items():
                if not estimator.startswith("single_micro_m"):
                    continue
                microbatches = int(estimator.rsplit("m", 1)[1])
                merged = double.merge(
                    candidate[["seed", "repetition", "mse"]].rename(
                        columns={"mse": "mse_single"}
                    ),
                    on=["seed", "repetition"],
                )
                ratio, ratio_lower, ratio_upper = bootstrap_ratio_ci(
                    merged["mse_single"].to_numpy(),
                    merged["mse_double"].to_numpy(),
                    bootstrap_repetitions,
                    seed,
                )
                if microbatches == 2:
                    passed = ratio_lower >= 0.95 and ratio_upper <= 1.05
                    reason = "checkpoint M=2 MSE equivalence"
                else:
                    passed = ratio_upper < 1
                    reason = "checkpoint M>2 lower MSE"
                decisions.append(
                    {
                        **common,
                        "hypothesis": "H3",
                        "estimator": estimator,
                        "microbatches": microbatches,
                        "estimate": ratio,
                        "ci_lower": ratio_lower,
                        "ci_upper": ratio_upper,
                        "raw_pvalue": np.nan,
                        "pass": bool(passed),
                        "reason": reason,
                    }
                )
    return decisions, summaries


def _continual_analysis(frame: pd.DataFrame) -> list[dict[str, Any]]:
    final = frame[frame["phase"] == "final"].copy()
    return (
        final.groupby(["scenario", "method", "strength"])
        .agg(
            final_average_accuracy=("final_average_accuracy", "mean"),
            final_accuracy_std=("final_average_accuracy", "std"),
            average_forgetting=("average_forgetting", "mean"),
            backward_transfer=("backward_transfer", "mean"),
            elapsed_seconds=("elapsed_seconds", "mean"),
            peak_memory_bytes=("peak_memory_bytes", "max"),
        )
        .reset_index()
        .assign(source="continual")
        .to_dict("records")
    )


def _apply_holm(decisions: list[dict[str, Any]]) -> None:
    indices = [
        index
        for index, row in enumerate(decisions)
        if row.get("applicable", True) and np.isfinite(row.get("raw_pvalue", np.nan))
    ]
    if not indices:
        return
    adjusted = multipletests(
        [decisions[index]["raw_pvalue"] for index in indices],
        method="holm",
    )[1]
    for index, value in zip(indices, adjusted):
        decisions[index]["holm_pvalue"] = float(value)
        if decisions[index]["hypothesis"] in {"H1", "H2"}:
            decisions[index]["pass"] = bool(
                decisions[index]["pass"] and value < 0.05
            )


def _plot_simulation(summary: pd.DataFrame, output: Path) -> None:
    if summary.empty:
        return
    sns.set_theme(style="whitegrid")
    chart = sns.relplot(
        data=summary,
        x="batch_size",
        y="mse",
        hue="estimator",
        col="population",
        kind="line",
        marker="o",
        facet_kws={"sharey": False},
    )
    chart.set(xscale="log", yscale="log")
    chart.figure.suptitle("Estimator MSE by batch size", y=1.03)
    chart.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(chart.figure)


def _write_report(
    path: Path,
    decisions: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    lines = [
        "# Parameter-importance validation report",
        "",
        "The decisions below are generated from the preregistered rules.",
        "",
        "## Hypothesis decisions",
        "",
    ]
    if not decisions:
        lines.append("No simulation decision data were found.")
    else:
        decision_frame = pd.DataFrame(decisions)
        for hypothesis in ["H1", "H2", "H3"]:
            subset = decision_frame[decision_frame["hypothesis"] == hypothesis]
            if subset.empty:
                continue
            applicable = subset[subset.get("applicable", True).astype(bool)]
            passed = int(applicable["pass"].sum())
            lines.append(
                f"- **{hypothesis}:** {passed}/{len(applicable)} applicable conditions passed"
                f" ({len(subset) - len(applicable)} stress-only conditions)."
            )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `decisions.parquet`: condition-level H1/H2/H3 decisions.",
            "- `summary.parquet`: estimator and continual-learning summaries.",
            "- `simulation-mse.png`: MSE comparison when simulation data are present.",
            "",
            f"Analyzed summary rows: {len(summaries)}.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(config: dict[str, Any], force: bool = False) -> Path:
    run_dir = prepare_run_dir(config, "analyze")
    inputs = discover_results([str(value) for value in config.get("inputs", ["outputs"])])
    if not inputs:
        raise FileNotFoundError("No results.parquet files found in analyze.inputs")
    write_metadata(run_dir, config, "analyze", "running", {"inputs": [str(p) for p in inputs]})

    bootstrap_repetitions = int(config.get("bootstrap_repetitions", 10_000))
    seed = int(config.get("seed", 0))
    decisions: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    simulation_summary: list[dict[str, Any]] = []
    for path in inputs:
        frame = pd.read_parquet(path)
        if {"naive", "double", "single_direct", "single_micro", "oracle"}.issubset(frame.columns):
            current_decisions, current_summary = _simulation_analysis(
                frame,
                bootstrap_repetitions,
                seed,
            )
            decisions.extend(current_decisions)
            summaries.extend(current_summary)
            simulation_summary.extend(current_summary)
        elif {"aggregation", "signed_bias", "quadrature"}.issubset(frame.columns):
            current_decisions, current_summary = _checkpoint_analysis(
                frame,
                bootstrap_repetitions,
                seed,
            )
            decisions.extend(current_decisions)
            summaries.extend(current_summary)
        elif {"phase", "final_average_accuracy", "method"}.issubset(frame.columns):
            summaries.extend(_continual_analysis(frame))

    _apply_holm(decisions)
    decisions_frame = pd.DataFrame(decisions)
    summaries_frame = pd.DataFrame(summaries)
    if not decisions_frame.empty:
        write_parquet(decisions_frame, run_dir / "decisions.parquet")
    write_parquet(summaries_frame, run_dir / "summary.parquet")
    _plot_simulation(pd.DataFrame(simulation_summary), run_dir / "simulation-mse.png")
    _write_report(run_dir / "report.md", decisions, summaries)
    (run_dir / "decision-summary.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    write_metadata(
        run_dir,
        config,
        "analyze",
        "completed",
        {
            "inputs": [str(path) for path in inputs],
            "decision_rows": len(decisions),
            "summary_rows": len(summaries),
        },
    )
    return run_dir
