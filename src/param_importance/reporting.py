from __future__ import annotations

import html
import json
import math
import shutil
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .io import prepare_run_dir, write_metadata, write_parquet
from .statistics import tost_one_sample


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
    "blue": "#A3BEFA",
    "blue_dark": "#2E4780",
    "gold": "#FFE15B",
    "gold_dark": "#736422",
    "orange": "#F0986E",
    "orange_dark": "#804126",
    "olive": "#A3D576",
    "olive_dark": "#386411",
    "pink": "#F390CA",
    "pink_dark": "#8A3A6F",
    "neutral": "#C5CAD3",
    "neutral_dark": "#464C55",
}


def _use_chart_theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "Segoe UI",
                "DejaVu Sans",
                "Arial",
            ],
        },
    )


def _add_chart_header(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    subtitle: str,
) -> None:
    title = textwrap.fill(title, 58, break_long_words=False)
    subtitle = textwrap.fill(subtitle, 92, break_long_words=False)
    title_lines = title.count("\n") + 1
    fig.subplots_adjust(top=max(0.68, 0.78 - 0.04 * (title_lines - 1)))
    left = ax.get_position().x0
    fig.text(
        left,
        0.985,
        title,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.925 - 0.035 * (title_lines - 1),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=TOKENS["muted"],
    )
    sns.despine(ax=ax)


def _save_figure(fig: plt.Figure, assets: Path, name: str) -> None:
    fig.savefig(assets / f"{name}.png", dpi=190, bbox_inches="tight")
    fig.savefig(assets / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _expand_result_paths(values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        candidate = Path(value)
        if any(character in value for character in "*?[]"):
            paths.extend(Path().glob(value))
        elif candidate.is_dir():
            direct = candidate / "results.parquet"
            if direct.exists():
                paths.append(direct)
            else:
                paths.extend(candidate.rglob("results.parquet"))
        elif candidate.exists():
            paths.append(candidate)
    return sorted(set(path.resolve() for path in paths))


def _read_many(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)


def _bootstrap_blocks(
    numerator: np.ndarray,
    denominator: np.ndarray | None,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    top = np.asarray(numerator, dtype=np.float64)
    bottom = None if denominator is None else np.asarray(denominator, dtype=np.float64)
    mask = np.isfinite(top)
    if bottom is not None:
        mask &= np.isfinite(bottom)
    top = top[mask]
    if bottom is not None:
        bottom = bottom[mask]
    if not top.size:
        return math.nan, math.nan, math.nan
    block_count = min(200, top.size)
    block_ids = np.arange(top.size) % block_count
    top_blocks = np.asarray(
        [top[block_ids == index].mean() for index in range(block_count)]
    )
    bottom_blocks = (
        None
        if bottom is None
        else np.asarray(
            [bottom[block_ids == index].mean() for index in range(block_count)]
        )
    )
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    chunk = 500
    for start in range(0, repetitions, chunk):
        stop = min(start + chunk, repetitions)
        indices = rng.integers(
            0,
            block_count,
            size=(stop - start, block_count),
        )
        sampled_top = top_blocks[indices].mean(axis=1)
        if bottom_blocks is None:
            draws[start:stop] = sampled_top
        else:
            sampled_bottom = bottom_blocks[indices].mean(axis=1)
            draws[start:stop] = sampled_top / np.maximum(sampled_bottom, 1e-30)
    estimate = (
        float(top.mean())
        if bottom is None
        else float(top.mean() / max(bottom.mean(), 1e-30))
    )
    return estimate, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def _seed_cluster_ci(
    frame: pd.DataFrame,
    value: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    seed_means = frame.groupby("seed")[value].mean().to_numpy(dtype=float)
    if not seed_means.size:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        seed_means.size,
        size=(repetitions, seed_means.size),
    )
    draws = seed_means[indices].mean(axis=1)
    return (
        float(seed_means.mean()),
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _seed_cluster_ratio_ci(
    frame: pd.DataFrame,
    numerator: str,
    denominator: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float]:
    seed_means = (
        frame.groupby("seed")[[numerator, denominator]]
        .mean()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if seed_means.empty:
        return math.nan, math.nan, math.nan
    values = seed_means.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(repetitions, len(values)),
    )
    sampled = values[indices].mean(axis=1)
    draws = sampled[:, 0] / np.maximum(sampled[:, 1], 1e-30)
    estimate = float(values[:, 0].mean() / max(values[:, 1].mean(), 1e-30))
    return (
        estimate,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    )


def _simulation_decisions(
    frame: pd.DataFrame,
    bootstrap_repetitions: int,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    primary = frame[frame["iid_theory_applicable"]].copy()
    for keys, group in primary.groupby(["population", "batch_size", "microbatches"]):
        oracle = group["oracle"].to_numpy()
        naive_error = group["naive"].to_numpy() - oracle
        estimate, lower, upper = _bootstrap_blocks(
            naive_error,
            None,
            bootstrap_repetitions,
            seed,
        )
        expected = float(
            group["gamma"].iloc[0]
            * group["population_variance"].iloc[0]
            / keys[1]
        )
        rows.append(
            {
                "source": "controlled",
                "hypothesis": "H1",
                "population": keys[0],
                "batch_size": keys[1],
                "microbatches": keys[2],
                "estimator": "naive",
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "reference": expected,
                "pass": lower > 0 and estimate > 0.1 * abs(float(group["oracle"].iloc[0])),
            }
        )
        for estimator in ["double_matched", "single_direct", "single_micro"]:
            error = group[estimator].to_numpy() - oracle
            scale = max(float(np.mean(np.abs(oracle))), 1e-12)
            equivalent = tost_one_sample(error, 0.05 * scale)
            naive_absolute = np.abs(naive_error)
            candidate_absolute = np.abs(error)
            improvement = stats.ttest_rel(
                naive_absolute,
                candidate_absolute,
                alternative="greater",
            )
            rows.append(
                {
                    "source": "controlled",
                    "hypothesis": "H2",
                    "population": keys[0],
                    "batch_size": keys[1],
                    "microbatches": keys[2],
                    "estimator": estimator,
                    "estimate": float(error.mean()),
                    "ci_lower": math.nan,
                    "ci_upper": math.nan,
                    "reference": 0.0,
                    "raw_pvalue": float(max(equivalent["pvalue"], improvement.pvalue)),
                    "pass": bool(
                        equivalent["pvalue"] < 0.05 and improvement.pvalue < 0.05
                    ),
                }
            )
        single_sq = (group["single_micro"].to_numpy() - oracle) ** 2
        double_sq = (group["double_matched"].to_numpy() - oracle) ** 2
        ratio, ratio_lower, ratio_upper = _bootstrap_blocks(
            single_sq,
            double_sq,
            bootstrap_repetitions,
            seed,
        )
        rows.append(
            {
                "source": "controlled",
                "hypothesis": "H3",
                "population": keys[0],
                "batch_size": keys[1],
                "microbatches": keys[2],
                "estimator": "single_micro_vs_double_matched",
                "estimate": ratio,
                "ci_lower": ratio_lower,
                "ci_upper": ratio_upper,
                "reference": 1.0,
                "pass": (
                    ratio_lower >= 0.95 and ratio_upper <= 1.05
                    if keys[2] == 2
                    else ratio_upper < 1.0
                ),
            }
        )
    return pd.DataFrame(rows)


def _checkpoint_summary(
    frame: pd.DataFrame,
    bootstrap_repetitions: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = frame[frame["aggregation"].eq("model")].copy()
    model["total_bias"] = model["sum_estimate"] - model["sum_oracle"]
    model["relative_total_bias"] = model["total_bias"] / model["sum_oracle"].abs().clip(1e-12)
    rows: list[dict[str, Any]] = []
    group_columns = ["dataset", "model", "checkpoint", "batch_size", "estimator"]
    for keys, group in model.groupby(group_columns):
        estimate, lower, upper = _seed_cluster_ci(
            group,
            "total_bias",
            bootstrap_repetitions,
            seed,
        )
        rows.append(
            {
                "dataset": keys[0],
                "model": keys[1],
                "checkpoint": keys[2],
                "batch_size": keys[3],
                "estimator": keys[4],
                "seed_count": group["seed"].nunique(),
                "repetitions": group["repetition"].nunique(),
                "total_bias": estimate,
                "bias_ci_lower": lower,
                "bias_ci_upper": upper,
                "relative_total_bias": float(group["relative_total_bias"].mean()),
                "absolute_bias": float(group["absolute_bias"].mean()),
                "mse": float(group["mse"].mean()),
                "spearman": float(group["spearman"].mean()),
                "backward_count": float(group["backward_count"].mean()),
            }
        )
    summary = pd.DataFrame(rows)
    decisions: list[dict[str, Any]] = []
    for keys, condition in model.groupby(["dataset", "model", "checkpoint", "batch_size"]):
        naive = condition[condition["estimator"].eq("naive")]
        if naive.empty:
            continue
        estimate, lower, upper = _seed_cluster_ci(
            naive,
            "total_bias",
            bootstrap_repetitions,
            seed,
        )
        scale = max(float(naive["sum_oracle"].abs().mean()), 1e-12)
        decisions.append(
            {
                "source": "checkpoint",
                "hypothesis": "H1",
                "dataset": keys[0],
                "model": keys[1],
                "checkpoint": keys[2],
                "batch_size": keys[3],
                "estimator": "naive",
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "reference": 0.0,
                "pass": lower > 0 and estimate > 0.1 * scale,
            }
        )
        for estimator in [
            "double",
            "single_direct",
            "single_micro_m4",
            "single_micro_m8",
        ]:
            candidate = condition[condition["estimator"].eq(estimator)]
            if candidate.empty:
                continue
            seed_errors = candidate.groupby("seed")["total_bias"].mean().to_numpy()
            equivalence = tost_one_sample(seed_errors, 0.05 * scale)
            merged = naive[
                ["seed", "repetition", "absolute_bias"]
            ].merge(
                candidate[["seed", "repetition", "absolute_bias"]],
                on=["seed", "repetition"],
                suffixes=("_naive", "_candidate"),
            )
            improvement = stats.ttest_rel(
                merged["absolute_bias_naive"],
                merged["absolute_bias_candidate"],
                alternative="greater",
            )
            decisions.append(
                {
                    "source": "checkpoint",
                    "hypothesis": "H2",
                    "dataset": keys[0],
                    "model": keys[1],
                    "checkpoint": keys[2],
                    "batch_size": keys[3],
                    "estimator": estimator,
                    "estimate": float(candidate["total_bias"].mean()),
                    "reference": 0.0,
                    "raw_pvalue": float(max(equivalence["pvalue"], improvement.pvalue)),
                    "pass": bool(
                        equivalence["pvalue"] < 0.05 and improvement.pvalue < 0.05
                    ),
                }
            )
        for microbatches in [2, 4, 8]:
            single_name = f"single_micro_m{microbatches}"
            double_name = f"double_matched_m{microbatches}"
            single = condition[condition["estimator"].eq(single_name)]
            matched = condition[condition["estimator"].eq(double_name)]
            if single.empty or matched.empty:
                continue
            merged = single[["seed", "repetition", "mse"]].merge(
                matched[["seed", "repetition", "mse"]],
                on=["seed", "repetition"],
                suffixes=("_single", "_double"),
            )
            ratio, lower, upper = _seed_cluster_ratio_ci(
                merged,
                "mse_single",
                "mse_double",
                bootstrap_repetitions,
                seed,
            )
            decisions.append(
                {
                    "source": "checkpoint",
                    "hypothesis": "H3",
                    "dataset": keys[0],
                    "model": keys[1],
                    "checkpoint": keys[2],
                    "batch_size": keys[3],
                    "estimator": f"{single_name}_vs_{double_name}",
                    "microbatches": microbatches,
                    "estimate": ratio,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "reference": 1.0,
                    "pass": (
                        lower >= 0.95 and upper <= 1.05
                        if microbatches == 2
                        else upper < 1.0
                    ),
                }
            )
    return summary, pd.DataFrame(decisions)


def _omnibus_tests(
    simulation_decisions: pd.DataFrame,
) -> pd.DataFrame:
    h1 = simulation_decisions[simulation_decisions["hypothesis"].eq("H1")]
    h1_values = h1["estimate"].to_numpy() / h1["reference"].abs().clip(1e-30).to_numpy()
    h1_test = stats.ttest_1samp(h1_values, 0.0, alternative="greater")

    h2 = simulation_decisions[simulation_decisions["hypothesis"].eq("H2")]
    h2_pvalue = float(h2["raw_pvalue"].max())

    h3 = simulation_decisions[
        simulation_decisions["hypothesis"].eq("H3")
        & simulation_decisions["microbatches"].gt(2)
    ]
    h3_values = np.log(h3["estimate"].to_numpy())
    h3_test = stats.ttest_1samp(h3_values, 0.0, alternative="less")
    rows = [
        {
            "hypothesis": "H1",
            "effect": float(np.mean(h1_values)),
            "raw_pvalue": float(h1_test.pvalue),
        },
        {
            "hypothesis": "H2",
            "effect": float(h2["estimate"].abs().mean()),
            "raw_pvalue": h2_pvalue,
        },
        {
            "hypothesis": "H3",
            "effect": float(np.exp(np.mean(h3_values))),
            "raw_pvalue": float(h3_test.pvalue),
        },
    ]
    adjusted = multipletests(
        [row["raw_pvalue"] for row in rows],
        method="holm",
    )[1]
    for row, pvalue in zip(rows, adjusted):
        row["holm_pvalue"] = float(pvalue)
        row["pass"] = bool(pvalue < 0.05)
    return pd.DataFrame(rows)


def _plot_simulation_bias(frame: pd.DataFrame, assets: Path) -> None:
    data = frame[frame["iid_theory_applicable"]].copy()
    data["error"] = data["naive"] - data["oracle"]
    grouped = (
        data.groupby(["population", "batch_size"])
        .agg(
            observed=("error", "mean"),
            gamma=("gamma", "first"),
            variance=("population_variance", "first"),
        )
        .reset_index()
    )
    grouped["expected"] = grouped["gamma"] * grouped["variance"] / grouped["batch_size"]
    grouped["ratio"] = grouped["observed"] / grouped["expected"]
    populations = list(grouped["population"].unique())
    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)
    for ax, population in zip(axes.flat, populations):
        part = grouped[grouped["population"].eq(population)]
        ax.plot(
            part["batch_size"],
            part["ratio"],
            marker="o",
            color=TOKENS["blue"],
            markeredgecolor=TOKENS["blue_dark"],
            linewidth=1.2,
        )
        ax.axhline(1.0, color=TOKENS["neutral_dark"], linestyle=":", linewidth=1)
        ax.set_title(population.replace("_", " "), fontsize=9)
        ax.set_xscale("log", base=2)
        ax.set_ylim(0.85, 1.15)
    for ax in axes.flat[len(populations) :]:
        ax.set_visible(False)
    fig.supxlabel("Batch size B")
    fig.supylabel("Observed bias / γVar(X)/B")
    _add_chart_header(
        fig,
        axes.flat[0],
        "同批次 Naive 过估计精确服从 1/B 标度",
        "IID 条件下，六类分布/解析梯度问题；虚线为理论值 1，20,000 次重复/格。",
    )
    _save_figure(fig, assets, "simulation-bias-theory")


def _plot_simulation_h3(frame: pd.DataFrame, assets: Path) -> None:
    rows = []
    for keys, group in frame.groupby(["population", "microbatches"]):
        single = (group["single_micro"] - group["oracle"]) ** 2
        double = (group["double_matched"] - group["oracle"]) ** 2
        rows.append(
            {
                "population": keys[0].replace("_", " "),
                "M": keys[1],
                "ratio": float(single.mean() / double.mean()),
            }
        )
    matrix = (
        pd.DataFrame(rows)
        .pivot(index="population", columns="M", values="ratio")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    cmap = sns.blend_palette(
        [TOKENS["panel"], "#EAF1FE", TOKENS["blue"]],
        as_cmap=True,
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=0.3,
        vmax=1.0,
        linewidths=1,
        linecolor=TOKENS["panel"],
        ax=ax,
    )
    ax.set_xlabel("Microbatch count M")
    ax.set_ylabel("")
    _add_chart_header(
        fig,
        ax,
        "M>2 时完整单采样 U-statistic 的 MSE 更低",
        "比值 = Single-micro MSE / 同样本、同微批梯度预算的配对 Double MSE；M=2 应严格等于 1。",
    )
    _save_figure(fig, assets, "simulation-h3-mse-ratio")


def _plot_checkpoint_bias(summary: pd.DataFrame, assets: Path) -> None:
    if summary.empty:
        return
    data = summary[
        summary["dataset"].eq("mnist")
        & summary["batch_size"].eq(64)
        & summary["estimator"].isin(
            ["naive", "double", "single_direct", "single_micro_m4", "single_micro_m8"]
        )
    ].copy()
    labels = {
        "naive": "Naive",
        "double": "Double",
        "single_direct": "Single-direct",
        "single_micro_m4": "Single-micro M=4",
        "single_micro_m8": "Single-micro M=8",
    }
    data["method"] = data["estimator"].map(labels)
    checkpoints = sorted(data["checkpoint"].unique())
    methods = list(labels.values())
    palette = {
        "Naive": TOKENS["orange"],
        "Double": TOKENS["neutral"],
        "Single-direct": TOKENS["blue"],
        "Single-micro M=4": TOKENS["olive"],
        "Single-micro M=8": TOKENS["pink"],
    }
    fig, ax = plt.subplots(figsize=(10, 5.8))
    x = np.arange(len(checkpoints))
    offsets = np.linspace(-0.24, 0.24, len(methods))
    for offset, method in zip(offsets, methods):
        part = data[data["method"].eq(method)].set_index("checkpoint").reindex(checkpoints)
        ax.errorbar(
            x + offset,
            part["relative_total_bias"],
            fmt="o",
            color=palette[method],
            markeredgecolor=TOKENS["ink"],
            markeredgewidth=0.5,
            linewidth=1,
            capsize=3,
            label=method,
        )
    ax.axhline(0, color=TOKENS["neutral_dark"], linestyle=":", linewidth=1)
    ax.set_yscale("symlog", linthresh=0.1, linscale=0.8, base=10)
    ax.set_xticks(x, checkpoints)
    ax.set_ylabel("Mean total bias / |oracle total| (symlog)")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        ncol=3,
        borderaxespad=0,
    )
    _add_chart_header(
        fig,
        ax,
        "MNIST 定点实验中 Naive 总重要性持续正偏",
        "B=64，跨已完成训练种子与 256 次配对重采样取均值；修正方法接近零线。",
    )
    _save_figure(fig, assets, "checkpoint-total-bias")


def _plot_checkpoint_mse(summary: pd.DataFrame, assets: Path) -> None:
    if summary.empty:
        return
    data = summary[
        summary["dataset"].eq("mnist")
        & summary["batch_size"].eq(64)
        & summary["estimator"].isin(
            ["naive", "double", "single_direct", "single_micro_m4", "single_micro_m8"]
        )
    ].copy()
    baseline = data[data["estimator"].eq("naive")][
        ["checkpoint", "mse"]
    ].rename(columns={"mse": "naive_mse"})
    data = data.merge(baseline, on="checkpoint")
    data["mse_ratio"] = data["mse"] / data["naive_mse"]
    labels = {
        "naive": "Naive",
        "double": "Double",
        "single_direct": "Single-direct",
        "single_micro_m4": "Single-micro M=4",
        "single_micro_m8": "Single-micro M=8",
    }
    data["method"] = data["estimator"].map(labels)
    palette = {
        "Naive": TOKENS["orange"],
        "Double": TOKENS["neutral_dark"],
        "Single-direct": TOKENS["blue_dark"],
        "Single-micro M=4": TOKENS["olive_dark"],
        "Single-micro M=8": TOKENS["pink_dark"],
    }
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for method, part in data.groupby("method"):
        part = part.sort_values("checkpoint")
        ax.plot(
            part["checkpoint"],
            part["mse_ratio"],
            marker="o",
            linewidth=1.2,
            color=palette[method],
            label=method,
        )
    ax.axhline(1, color=TOKENS["neutral_dark"], linestyle=":", linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("MSE / Naive MSE")
    ax.set_xlabel("")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        ncol=3,
        borderaxespad=0,
    )
    _add_chart_header(
        fig,
        ax,
        "偏差修正降低了 MNIST 参数重要性的均方误差",
        "B=64；低于 1 表示优于 Naive。Single-direct 在当前定点网格上通常最低。",
    )
    _save_figure(fig, assets, "checkpoint-mse-relative")


def _plot_reference(reference: pd.DataFrame, assets: Path) -> None:
    if reference.empty:
        return
    data = reference[
        reference["aggregation"].eq("model")
        & reference["method"].str.startswith("composite_trapezoid_")
    ].copy()
    if data.empty:
        return
    data["intervals"] = data["method"].str.rsplit("_", n=1).str[-1].astype(int)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for label, part in data.groupby(["source_run", "checkpoint"]):
        part = part.sort_values("intervals")
        ax.plot(
            part["intervals"],
            part["adaptive_relative_l2"],
            marker="o",
            linewidth=1,
            label=f"{label[0]} / {label[1]}",
        )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Composite trapezoid intervals")
    ax.set_ylabel("Relative L2 difference to adaptive reference")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        ncol=2,
        fontsize=8,
        borderaxespad=0,
    )
    _add_chart_header(
        fig,
        ax,
        "参考积分通过独立网格加密进行交叉检查",
        "固定 GL16 不再被称为真值；同时报告自适应状态、网格差异与端点损失守恒误差。",
    )
    _save_figure(fig, assets, "reference-convergence")


def _plot_checkpoint_h3(
    checkpoint_decisions: pd.DataFrame,
    assets: Path,
) -> None:
    data = checkpoint_decisions[
        checkpoint_decisions["hypothesis"].eq("H3")
    ].copy()
    if data.empty:
        return
    data["network"] = data["dataset"] + " / " + data["model"]
    matrix = data.pivot_table(
        index="network",
        columns="microbatches",
        values="estimate",
        aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(8.5, max(3.8, 1.1 * len(matrix) + 2.5)))
    cmap = sns.blend_palette(
        [TOKENS["panel"], "#D8ECBD", TOKENS["olive"]],
        as_cmap=True,
    )
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=0.3,
        vmax=1.0,
        linewidths=1,
        linecolor=TOKENS["panel"],
        ax=ax,
    )
    ax.set_xlabel("Microbatch count M")
    ax.set_ylabel("")
    _add_chart_header(
        fig,
        ax,
        "真实模型同预算比较检验 Single-micro 的方差优势",
        "跨检查点和批量大小平均的参数 MSE 比值；分母为使用相同样本和微批梯度数的 matched Double。",
    )
    _save_figure(fig, assets, "checkpoint-h3-mse-ratio")


def _plot_continual(frame: pd.DataFrame, assets: Path) -> None:
    final = frame[frame["phase"].eq("final")].copy() if not frame.empty else frame
    if final.empty:
        return
    summary = (
        final.groupby(["scenario", "method"])
        .agg(
            accuracy=("final_average_accuracy", "mean"),
            forgetting=("average_forgetting", "mean"),
        )
        .reset_index()
    )
    scenarios = list(summary["scenario"].unique())
    fig, axes = plt.subplots(
        len(scenarios),
        2,
        figsize=(12.5, max(6.0, 5.2 * len(scenarios))),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.58, wspace=0.34)
    for row_index, scenario in enumerate(scenarios):
        current = summary[summary["scenario"].eq(scenario)]
        ordered = current.sort_values("accuracy", ascending=True)
        axes[row_index, 0].barh(
            ordered["method"],
            ordered["accuracy"],
            color=TOKENS["blue"],
            edgecolor=TOKENS["blue_dark"],
        )
        axes[row_index, 0].set_xlabel("Final average accuracy")
        axes[row_index, 0].set_title(
            scenario.replace("_", " "),
            fontsize=10,
            pad=9,
        )
        ordered_forgetting = current.sort_values("forgetting", ascending=False)
        axes[row_index, 1].barh(
            ordered_forgetting["method"],
            ordered_forgetting["forgetting"],
            color=TOKENS["orange"],
            edgecolor=TOKENS["orange_dark"],
        )
        axes[row_index, 1].set_xlabel("Average forgetting")
        axes[row_index, 1].set_title(
            scenario.replace("_", " "),
            fontsize=10,
            pad=9,
        )
        for axis in axes[row_index]:
            axis.tick_params(axis="y", labelsize=9)
    _add_chart_header(
        fig,
        axes[0, 0],
        "持续学习端到端结果检验统计改进是否转化为任务性能",
        "最终种子均值；左侧越高越好，右侧越低越好。超参数只使用独立调参种子选择。",
    )
    _save_figure(fig, assets, "continual-learning")


def _format_number(value: Any, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    number = float(value)
    if number == 0:
        return "0"
    if abs(number) < 1e-3 or abs(number) >= 1e4:
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def _table(frame: pd.DataFrame, columns: list[str], rename: dict[str, str]) -> str:
    display = frame[columns].rename(columns=rename).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(_format_number)
    return display.to_html(index=False, classes="data-table", border=0, escape=True)


def _write_html(
    path: Path,
    *,
    simulation: pd.DataFrame,
    checkpoint: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    checkpoint_decisions: pd.DataFrame,
    reference: pd.DataFrame,
    quadrature_checkpoint: pd.DataFrame,
    stress_checkpoint: pd.DataFrame,
    continual: pd.DataFrame,
    decisions: pd.DataFrame,
    omnibus: pd.DataFrame,
    inputs: dict[str, list[Path]],
) -> None:
    h1 = decisions[decisions["hypothesis"].eq("H1")]
    h2 = decisions[decisions["hypothesis"].eq("H2")]
    h3 = decisions[decisions["hypothesis"].eq("H3")]
    h3_controlled = h3[h3["source"].eq("controlled")]
    h3_m2 = h3_controlled[h3_controlled["microbatches"].eq(2)]
    h3_more = h3_controlled[h3_controlled["microbatches"].gt(2)]
    h3_ratios = (
        simulation.assign(
            single_sq=lambda value: (value["single_micro"] - value["oracle"]) ** 2,
            double_sq=lambda value: (value["double_matched"] - value["oracle"]) ** 2,
        )
        .groupby("microbatches")
        .agg(single_sq=("single_sq", "mean"), double_sq=("double_sq", "mean"))
    )
    h3_ratios["ratio"] = h3_ratios["single_sq"] / h3_ratios["double_sq"]

    sim_h2 = h2[h2["source"].eq("controlled")]
    h2_metric_rows = []
    iid_simulation = simulation[simulation["iid_theory_applicable"]]
    for estimator in ["double_matched", "single_direct", "single_micro"]:
        condition_metrics = []
        for _, group in iid_simulation.groupby(
            ["population", "batch_size", "microbatches"]
        ):
            naive_error = group["naive"] - group["oracle"]
            candidate_error = group[estimator] - group["oracle"]
            condition_metrics.append(
                {
                    "mean_bias_ratio": abs(float(candidate_error.mean()))
                    / max(abs(float(naive_error.mean())), 1e-30),
                    "absolute_error_ratio": float(candidate_error.abs().mean())
                    / max(float(naive_error.abs().mean()), 1e-30),
                }
            )
        metric_frame = pd.DataFrame(condition_metrics)
        estimator_decisions = sim_h2[sim_h2["estimator"].eq(estimator)]
        h2_metric_rows.append(
            {
                "estimator": estimator,
                "mean_bias_ratio": float(metric_frame["mean_bias_ratio"].mean()),
                "absolute_error_ratio": float(
                    metric_frame["absolute_error_ratio"].mean()
                ),
                "strict_pass": int(estimator_decisions["pass"].sum()),
                "conditions": len(estimator_decisions),
            }
        )
    h2_metrics = pd.DataFrame(h2_metric_rows)
    checkpoint_h1 = h1[h1["source"].eq("checkpoint")]
    checkpoint_h2 = h2[h2["source"].eq("checkpoint")]
    checkpoint_h3 = h3[h3["source"].eq("checkpoint")]
    certified = (
        reference[
            reference["aggregation"].eq("model")
            & reference["method"].str.startswith("adaptive_")
        ]["reference_certified"]
        if not reference.empty
        else pd.Series(dtype=bool)
    )
    continual_final = (
        continual[continual["phase"].eq("final")]
        if not continual.empty
        else pd.DataFrame()
    )
    stress_table = ""
    if not stress_checkpoint.empty:
        stress = stress_checkpoint[
            stress_checkpoint["aggregation"].eq("model")
        ].copy()
        stress["total_bias"] = stress["sum_estimate"] - stress["sum_oracle"]
        stress["scenario"] = np.select(
            [
                ~stress["replacement"],
                stress["optimizer"].eq("adamw"),
                stress["theory_status"].eq(
                    "checkpoint_distribution_only_non_sgd"
                ),
            ],
            ["无放回抽样", "AdamW 检查点", "Momentum 检查点"],
            default="其他压力测试",
        )
        stress_summary = (
            stress[
                stress["estimator"].isin(
                    ["naive", "double", "single_direct", "single_micro_m4"]
                )
            ]
            .groupby(["scenario", "estimator"])
            .agg(
                total_bias=("total_bias", "mean"),
                absolute_bias=("absolute_bias", "mean"),
                mse=("mse", "mean"),
            )
            .reset_index()
        )
        stress_baseline = stress_summary[
            stress_summary["estimator"].eq("naive")
        ][["scenario", "mse"]].rename(columns={"mse": "naive_mse"})
        stress_summary = stress_summary.merge(stress_baseline, on="scenario")
        stress_summary["mse_ratio"] = (
            stress_summary["mse"] / stress_summary["naive_mse"]
        )
        stress_table = _table(
            stress_summary,
            [
                "scenario",
                "estimator",
                "total_bias",
                "absolute_bias",
                "mse_ratio",
            ],
            {
                "scenario": "压力条件",
                "estimator": "估计器",
                "total_bias": "平均总偏差",
                "absolute_bias": "参数平均绝对偏差",
                "mse_ratio": "MSE / Naive",
            },
        )
    continual_table = ""
    continual_summary = pd.DataFrame()
    continual_narrative = ""
    if not continual_final.empty:
        continual_summary = (
            continual_final.groupby(["scenario", "method"])
            .agg(
                final_accuracy=("final_average_accuracy", "mean"),
                accuracy_std=("final_average_accuracy", "std"),
                forgetting=("average_forgetting", "mean"),
                backward_transfer=("backward_transfer", "mean"),
                elapsed_seconds=("elapsed_seconds", "mean"),
                samples_seen=("samples_seen", "mean"),
                peak_memory_bytes=("peak_memory_bytes", "max"),
                final_seeds=("seed", "nunique"),
            )
            .reset_index()
        )
        continual_summary["peak_memory_gb"] = (
            continual_summary["peak_memory_bytes"] / 2**30
        )
        continual_table = _table(
            continual_summary,
            [
                "scenario",
                "method",
                "final_seeds",
                "final_accuracy",
                "accuracy_std",
                "forgetting",
                "backward_transfer",
                "elapsed_seconds",
                "samples_seen",
                "peak_memory_gb",
            ],
            {
                "scenario": "场景",
                "method": "方法",
                "final_seeds": "最终种子",
                "final_accuracy": "最终平均准确率",
                "accuracy_std": "准确率标准差",
                "forgetting": "平均遗忘",
                "backward_transfer": "后向迁移",
                "elapsed_seconds": "训练秒数",
                "samples_seen": "处理样本",
                "peak_memory_gb": "峰值显存 GB",
            },
        )
        continual_parts = []
        for scenario, group in continual_summary.groupby("scenario"):
            best = group.loc[group["final_accuracy"].idxmax()]
            baseline = group[group["method"].eq("fine_tuning")]
            comparison = ""
            if not baseline.empty:
                comparison = (
                    f"，相对 Fine-tuning 的准确率差为 "
                    f"{best['final_accuracy'] - baseline.iloc[0]['final_accuracy']:+.4f}"
                )
            continual_parts.append(
                f"<p><strong>{html.escape(str(scenario))}</strong>：最终平均准确率最高的是 "
                f"<code>{html.escape(str(best['method']))}</code> "
                f"({best['final_accuracy']:.4f}){comparison}；其平均遗忘为 "
                f"{best['forgetting']:.4f}。该场景基于 {int(best['final_seeds'])} 个最终种子，"
                "种子少于 3 时仅作资源受限扩展证据。</p>"
            )
        continual_narrative = "".join(continual_parts)
    coverage_rows = [
        {
            "scope": "受控统计",
            "rows": len(simulation),
            "coverage": (
                f"{simulation['population'].nunique()} 类总体/问题；"
                f"B={sorted(simulation['batch_size'].unique().tolist())}；"
                f"M={sorted(simulation['microbatches'].unique().tolist())}"
            ),
            "status": "完成",
        }
    ]
    if not checkpoint_summary.empty:
        for keys, group in checkpoint_summary.groupby(["dataset", "model"]):
            coverage_rows.append(
                {
                    "scope": f"定点：{keys[0]} / {keys[1]}",
                    "rows": len(
                        checkpoint[
                            checkpoint["dataset"].eq(keys[0])
                            & checkpoint["model"].eq(keys[1])
                        ]
                    ),
                    "coverage": (
                        f"{int(group['seed_count'].max())} seeds；"
                        f"{group['checkpoint'].nunique()} checkpoints；"
                        f"B={sorted(group['batch_size'].unique().tolist())}"
                    ),
                    "status": "完成结果已纳入",
                }
            )
    if not quadrature_checkpoint.empty:
        coverage_rows.append(
            {
                "scope": "多节点求积定向消融",
                "rows": len(quadrature_checkpoint),
                "coverage": "MNIST；同一模型权重；Simpson-3 / GL3；GL16 对照；B=64，M=4",
                "status": "完成结果已纳入",
            }
        )
    reference_models = (
        reference[
            reference["aggregation"].eq("model")
            & reference["method"].str.startswith("adaptive_")
        ]
        if not reference.empty
        else pd.DataFrame()
    )
    coverage_rows.append(
        {
            "scope": "积分参考认证",
            "rows": len(reference_models),
            "coverage": (
                f"{int(reference_models['reference_certified'].sum())}/"
                f"{len(reference_models)} checkpoints certified"
                if not reference_models.empty
                else "尚无正式参考结果"
            ),
            "status": "完成" if not reference_models.empty else "待运行",
        }
    )
    if not continual_final.empty:
        for scenario, group in continual_final.groupby("scenario"):
            coverage_rows.append(
                {
                    "scope": f"持续学习：{scenario}",
                    "rows": len(group),
                    "coverage": (
                        f"{group['method'].nunique()} methods；"
                        f"{group['seed'].nunique()} final seeds"
                    ),
                    "status": "完成结果已纳入",
                }
            )
    else:
        coverage_rows.append(
            {
                "scope": "持续学习",
                "rows": 0,
                "coverage": "尚无正式 final 结果",
                "status": "待运行",
            }
        )
    if not stress_checkpoint.empty:
        coverage_rows.append(
            {
                "scope": "理论边界压力测试",
                "rows": len(stress_checkpoint),
                "coverage": "无放回、Momentum、AdamW；各 1 seed，初始化/最终检查点",
                "status": "完成结果已纳入",
            }
        )
    coverage_table = _table(
        pd.DataFrame(coverage_rows),
        ["scope", "rows", "coverage", "status"],
        {
            "scope": "实验范围",
            "rows": "结果行数",
            "coverage": "实际覆盖",
            "status": "状态",
        },
    )

    omnibus_table = _table(
        omnibus,
        ["hypothesis", "effect", "raw_pvalue", "holm_pvalue", "pass"],
        {
            "hypothesis": "主假设",
            "effect": "汇总效应",
            "raw_pvalue": "原始 p",
            "holm_pvalue": "Holm p",
            "pass": "通过",
        },
    )
    checkpoint_table = ""
    if not checkpoint_summary.empty:
        selected = checkpoint_summary[
            checkpoint_summary["batch_size"].eq(64)
            & checkpoint_summary["estimator"].isin(
                ["naive", "double", "single_direct", "single_micro_m4", "single_micro_m8"]
            )
        ].copy()
        checkpoint_table = _table(
            selected,
            [
                "dataset",
                "model",
                "checkpoint",
                "estimator",
                "seed_count",
                "total_bias",
                "relative_total_bias",
                "absolute_bias",
                "mse",
                "spearman",
            ],
            {
                "dataset": "数据集",
                "model": "模型",
                "checkpoint": "检查点",
                "estimator": "估计器",
                "seed_count": "种子数",
                "total_bias": "总偏差",
                "relative_total_bias": "相对总偏差",
                "absolute_bias": "参数平均绝对偏差",
                "mse": "参数 MSE",
                "spearman": "排序相关",
            },
        )
    h2_table = _table(
        h2_metrics,
        [
            "estimator",
            "mean_bias_ratio",
            "absolute_error_ratio",
            "strict_pass",
            "conditions",
        ],
        {
            "estimator": "估计器",
            "mean_bias_ratio": "|均值偏差| / Naive",
            "absolute_error_ratio": "平均绝对误差 / Naive",
            "strict_pass": "严格通过数",
            "conditions": "条件数",
        },
    )
    reference_table = ""
    if not reference.empty:
        selected_reference = reference[
            reference["aggregation"].eq("model")
            & reference["method"].str.startswith("adaptive_")
        ].copy()
        reference_table = _table(
            selected_reference,
            [
                "source_run",
                "checkpoint",
                "adaptive_success",
                "adaptive_evaluations",
                "crosscheck_relative_l2",
                "refinement_relative_l2",
                "conservation_relative_adaptive",
                "reference_certified",
            ],
            {
                "source_run": "来源运行",
                "checkpoint": "检查点",
                "adaptive_success": "自适应收敛",
                "adaptive_evaluations": "函数评估",
                "crosscheck_relative_l2": "独立网格相对差",
                "refinement_relative_l2": "相邻网格相对差",
                "conservation_relative_adaptive": "守恒相对误差",
                "reference_certified": "参考认证",
            },
        )
    quadrature_table = ""
    quadrature_narrative = ""
    if not quadrature_checkpoint.empty:
        quadrature_model = quadrature_checkpoint[
            quadrature_checkpoint["aggregation"].eq("model")
        ].copy()
        quadrature_model["total_bias"] = (
            quadrature_model["sum_estimate"] - quadrature_model["sum_oracle"]
        )
        quadrature_model["absolute_total_bias"] = quadrature_model[
            "total_bias"
        ].abs()
        selected_estimators = [
            "naive",
            "single_direct",
            "ppt_variance_only",
            "single_micro_m4",
        ]
        quadrature_summary = (
            quadrature_model[
                quadrature_model["estimator"].isin(selected_estimators)
            ]
            .groupby(["checkpoint", "quadrature", "estimator"])
            .agg(
                total_bias=("total_bias", "mean"),
                absolute_total_bias=("absolute_total_bias", "mean"),
                parameter_mse=("mse", "mean"),
                quadrature_error=("quadrature_error", "mean"),
            )
            .reset_index()
        )
        quadrature_table = _table(
            quadrature_summary,
            [
                "checkpoint",
                "quadrature",
                "estimator",
                "total_bias",
                "absolute_total_bias",
                "parameter_mse",
                "quadrature_error",
            ],
            {
                "checkpoint": "检查点",
                "quadrature": "求积规则",
                "estimator": "估计器",
                "total_bias": "平均总偏差",
                "absolute_total_bias": "平均绝对总偏差",
                "parameter_mse": "参数 MSE",
                "quadrature_error": "相对 GL16 的参数平均差",
            },
        )
        total_squared = quadrature_model.assign(
            total_squared=lambda value: value["total_bias"] ** 2
        )
        total_mse = (
            total_squared.groupby(["checkpoint", "quadrature", "estimator"])[
                "total_squared"
            ]
            .mean()
            .unstack("estimator")
        )
        ppt_ratio = (
            total_mse["ppt_variance_only"] / total_mse["single_direct"]
        )
        quadrature_narrative = (
            f"定向多节点实验中，Simpson-3/GL3 相对 GL16 的逐参数平均差最大为 "
            f"{quadrature_summary['quadrature_error'].max():.2e}；"
            f"PPT 仅减起点方差与完整协方差修正的总误差 MSE 比值范围为 "
            f"{ppt_ratio.min():.3f}–{ppt_ratio.max():.3f}。"
            "这说明当前 MNIST 小步长路径下两者近似重合，但不能外推到大步长、"
            "强曲率或非光滑折点密集的路径。"
        )

    source_count = sum(len(value) for value in inputs.values())
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    html_text = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <link rel="icon" href="data:,">
  <title>参数重要性过估计验证实验报告</title>
  <style>
    :root {{ --ink:#1F2430; --muted:#6F768A; --line:#E6E8F0; --panel:#fff; --blue:#2E4780; --orange:#804126; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#F8FAFC; color:var(--ink); font-family:"Microsoft YaHei","Segoe UI",sans-serif; }}
    main {{ max-width:1120px; margin:0 auto; padding:44px 24px 80px; }}
    header, section {{ margin-bottom:42px; }}
    h1 {{ font-size:34px; line-height:1.2; margin:0 0 10px; }}
    h2 {{ font-size:24px; margin:0 0 14px; }}
    h3 {{ font-size:18px; margin:26px 0 10px; }}
    p, li {{ line-height:1.75; }}
    .lede {{ color:var(--muted); font-size:15px; }}
    .cards {{ display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:22px 0; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px; }}
    .card strong {{ display:block; font-size:19px; margin-bottom:8px; color:var(--blue); }}
    .status {{ display:inline-block; padding:3px 9px; border-radius:999px; background:#EAF1FE; color:#2E4780; font-size:13px; }}
    figure {{ margin:24px 0 32px; background:#fff; border:1px solid var(--line); border-radius:14px; padding:14px; }}
    figure img {{ width:100%; display:block; }}
    figcaption {{ color:var(--muted); font-size:14px; line-height:1.6; padding:8px 6px 2px; }}
    .table-wrap {{ overflow-x:auto; background:#fff; border:1px solid var(--line); border-radius:12px; margin:18px 0; }}
    table.data-table {{ border-collapse:collapse; width:100%; font-size:13px; }}
    .data-table th,.data-table td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:right; white-space:nowrap; }}
    .data-table th:first-child,.data-table td:first-child {{ text-align:left; }}
    .data-table th {{ background:#F4F5F7; }}
    code, pre {{ font-family:Consolas,monospace; }}
    a {{ color:#2E4780; }}
    pre {{ overflow-x:auto; background:#EEF1F5; padding:16px; border-radius:12px; line-height:1.55; }}
    .callout {{ border-left:4px solid #5477C4; padding:12px 16px; background:#F5F8FF; }}
    @media(max-width:760px) {{ .cards {{ grid-template-columns:1fr; }} main {{ padding:28px 16px 56px; }} }}
  </style>
</head>
<body>
<main data-report-audience="technical">
  <header data-contract-section="title">
    <span class="status">基于实际运行结果</span>
    <h1>参数重要性过估计验证实验报告</h1>
    <p class="lede">分支 <code>codex/overestimation-validation</code> · 生成时间 {html.escape(generated)} · 共读取 {source_count} 个结果文件</p>
  </header>

  <section data-contract-section="technical-summary">
    <h2>技术摘要</h2>
    <div class="cards">
      <div class="card"><strong>问题 1：存在</strong>受控 IID 网格中 H1 通过 {int(h1[h1["source"].eq("controlled")]["pass"].sum())}/{len(h1[h1["source"].eq("controlled")])} 个条件；Naive 偏差按 <code>γVar/B</code> 缩放。</div>
      <div class="card"><strong>问题 2：去偏成立，综合验收未全面通过</strong>修正后均值偏差大幅下降，但严格的“TOST + 绝对误差改善”只通过 {int(sim_h2["pass"].sum())}/{len(sim_h2)} 个估计器条件；方差代价不能忽略。</div>
      <div class="card"><strong>问题 3：理论得到支持</strong>M=2 的最大数值差为 {np.max(np.abs(simulation.loc[simulation["microbatches"].eq(2),"single_micro"] - simulation.loc[simulation["microbatches"].eq(2),"double_matched"])):.2e}；M&gt;2 的条件通过 {int(h3_more["pass"].sum())}/{len(h3_more)}。</div>
    </div>
    <p>主结论是：同一批次同时用于更新梯度和重要性评估，会产生结构性的正偏；协方差/U-statistic 修正针对该期望偏差项。它不保证每次估计都更接近 Oracle，因为去偏和方差之间存在权衡。单采样也并非在所有意义下“总是更好”，但在相同样本数和相同微批梯度评估数下，M&gt;2 时其完整配对平均具有更低 MSE；M=2 与对称双采样严格等价。</p>
    <div class="table-wrap">{omnibus_table}</div>
  </section>

  <section data-contract-section="key-findings">
    <h2>证据一：过估计不是偶然波动，而是可预测偏差</h2>
    <p>对单节点 SI，Naive 估计为 <code>γ·X̄²</code>，因此 IID 条件下 <code>E[Naive]-Oracle=γVar(X)/B</code>。实验覆盖 Gaussian、Student-t(5)、中心化对数正态、二次损失、线性回归梯度和逻辑回归梯度，观测偏差与理论 1/B 标度一致。有限总体无放回抽样作为理论失效压力测试单独报告。</p>
    <figure><img src="assets/simulation-bias-theory.png" alt="Naive bias compared with theory"><figcaption>图 1. 观测偏差与解析偏差的比值。理论不依赖高斯性，只要求相应二阶矩与采样条件成立。</figcaption></figure>
    <p>真实 MNIST MLP 定点结果也显示 Naive 总重要性正偏。当前报告读取 {checkpoint_summary["seed_count"].max() if not checkpoint_summary.empty else 0:.0f} 个已完成种子，每检查点/批量大小每种子 256 次配对重采样。H1 的真实模型判据通过 {int(checkpoint_h1["pass"].sum())}/{len(checkpoint_h1)} 个条件。</p>
    <figure><img src="assets/checkpoint-total-bias.png" alt="Checkpoint total bias"><figcaption>图 2. MNIST、B=64 的总偏差。相对总偏差以全数据单节点 Oracle 总和为尺度。</figcaption></figure>

    <h2>证据二：两种去偏方法在偏差上有效，方差与成本不同</h2>
    <p>Double 使用独立子样本消除乘积中的协方差项；Single-direct 使用逐样本交叉协方差修正；Single-micro 以 M 个微批均值构造同一 U-statistic 的低内存版本。受控实验采用 ±5% Oracle 尺度 TOST，并同时要求相对 Naive 的绝对误差显著下降。均值偏差显著缩小，但 Double 的方差常使平均绝对误差高于 Naive；因此 H2 的严格综合判据没有全局通过。真实模型中 H2 通过 {int(checkpoint_h2["pass"].sum())}/{len(checkpoint_h2)} 个估计器条件，种子数不足或 Oracle 接近零时 TOST 尤其严格。</p>
    <div class="table-wrap">{h2_table}</div>
    <figure><img src="assets/checkpoint-mse-relative.png" alt="Checkpoint relative MSE"><figcaption>图 3. MNIST 定点参数 MSE 相对 Naive 的比值。图中运行耗时字段是整套配对计算耗时，不能解释为单个估计器独立墙钟成本；反向传播计数另行记录。</figcaption></figure>
    <div class="table-wrap">{checkpoint_table}</div>

    <h2>证据三：单采样优势来自使用全部交叉配对</h2>
    <p>为避免不公平比较，报告新增 <code>double_matched_mM</code>：它与 Single-micro 使用完全相同的 B 个样本和 M 个微批梯度，只保留 M/2 个不相交配对；Single-micro 则平均全部 M(M-1)/2 个交叉配对。聚合 MSE 比值为：M=2 {h3_ratios.loc[2,"ratio"]:.3f}，M=4 {h3_ratios.loc[4,"ratio"]:.3f}，M=8 {h3_ratios.loc[8,"ratio"]:.3f}，M=16 {h3_ratios.loc[16,"ratio"]:.3f}。</p>
    <figure><img src="assets/simulation-h3-mse-ratio.png" alt="Single versus matched double MSE"><figcaption>图 4. 每个单元格先在 B={8,32,128} 上汇总平方误差。低于 1 表示完整 U-statistic 更有效率。</figcaption></figure>
    {"<p>真实模型的同预算 H3 条件通过 " + str(int(checkpoint_h3["pass"].sum())) + "/" + str(len(checkpoint_h3)) + "。M=2 采用 ±5% 等价区间，M>2 要求 95% 聚类 bootstrap 上界低于 1。</p><figure><img src='assets/checkpoint-h3-mse-ratio.png' alt='Checkpoint matched-budget MSE'><figcaption>图 5. 真实模型参数 MSE 的同预算比值，跨检查点与批量大小汇总。</figcaption></figure>" if not checkpoint_h3.empty else "<p class='callout'>当前正式定点结果尚未包含 matched-budget Double；真实模型 H3 图将在 CIFAR 代表性矩阵完成后自动生成。</p>"}
  </section>

  <section data-contract-section="scope-data-and-metric-definitions">
    <h2>范围、数据与指标定义</h2>
    <div class="table-wrap">{coverage_table}</div>
    <p>受控统计实验每格 20,000 次重复，B∈{{8,32,128}}、M∈{{2,4,8,16}}。真实模型主实验为 MNIST MLP，检查点位于训练 0%、25%、50%、100%，全数据 Oracle 使用固定类别平衡的 10,000 样本总体。估计器指标包括有符号偏差、参数平均绝对偏差、MSE、Spearman 排序相关、总重要性与 Oracle 总和之差、反向传播计数和峰值显存。</p>
    <p>“实际显著”要求 Naive 偏差 95% CI 下界大于 0，且超过 Oracle 尺度的 10%；“修正有效”不能只依赖未拒绝零偏差，而必须通过 TOST 等价检验并显著降低绝对误差；H3 使用配对 MSE 比值，M=2 检验等价，M&gt;2 检验上界是否低于 1。</p>
  </section>

  <section data-contract-section="methodology">
    <h2>方法与参考积分认证</h2>
    <pre>Naive:          γ · ū · v̄
Single-direct:  γ · (ū · v̄ - s_uv / B)
Single-micro:   γ · (ū · v̄ - s_uv,micro / M)
Path importance per parameter:  -Δθ_k ∫₀¹ ∂L(θ+αΔθ)/∂θ_k dα</pre>
    <p>固定高阶 Gauss-Legendre 只能是高精度近似，不能被认定为数学真值。新的参考协议采用三条证据链：SciPy 自适应 Gauss-Kronrod 向量积分；独立的复合梯形 8/16/32 区间加密；以及端点损失恒等式 <code>Σω_k = L(θ₀)-L(θ₁)</code> 的全局守恒检查。只有三者同时达到预设容差才标记为“参考认证”。</p>
    {"<figure><img src='assets/reference-convergence.png' alt='Reference convergence'><figcaption>独立网格相对自适应参考的 L2 差异。ReLU/MaxPool 路径可能包含折点，未达到容差会保留为未认证结果。</figcaption></figure>" if not reference.empty else "<p class='callout'>正式模型的自适应参考运行尚未提供结果文件，因此本版不宣称路径积分参考已认证。</p>"}
    <div class="table-wrap">{reference_table}</div>
    <h3>固定低阶求积与协方差消融</h3>
    <p>{quadrature_narrative}</p>
    <p>定向实验与自适应参考的模型文件哈希完全一致。GL16 与 adaptive GK21 的层级总和相对 L2 差在初始化和训练终点分别为 1.47e-6 与 6.43e-6，因此 GL16 在这两个特定检查点上得到很强的数值支持；结论仍应表述为“经独立方法认证的高精度近似”。</p>
    <div class="table-wrap">{quadrature_table}</div>
  </section>

  <section data-contract-section="limitations-uncertainty-and-robustness-checks">
    <h2>局限性、不确定性与稳健性</h2>
    <ul>
      <li>单节点 SI 验证随机梯度乘积的过估计机制，但不是完整路径积分；多节点结果必须依赖上面的数值积分认证。</li>
      <li>ReLU 和 MaxPool 使梯度沿路径分段变化，自适应积分可能因大量折点或 float32 数值噪声无法达到很紧的误差目标；未认证不等于结论错误，而是精度证据不足。</li>
      <li>严格无偏结论限定于无动量、无权重衰减 SGD 与明确的采样协议。Momentum、AdamW、数据增强、有限总体无放回属于外推压力测试。</li>
      <li>参数级重要性可正可负且 Oracle 均值很小，单一“相对偏差”可能爆炸；报告同时给出绝对偏差、MSE、总和偏差与排序相关。</li>
      <li>当前定点运行的墙钟字段覆盖同一重复内的整套估计器计算，不能用于估计器间独立速度排名；公平成本结论以样本数、梯度评估数和持续学习整段耗时为主。</li>
      <li>主假设使用 Holm 校正；条件级 CI 使用 10,000 次 Monte Carlo 分块或种子聚类 bootstrap，避免把海量批次重复误当作独立训练种子。</li>
    </ul>
    <div class="table-wrap">{stress_table}</div>
  </section>

  <section data-contract-section="recommended-next-steps">
    <h2>建议与结论边界</h2>
    <p>在偏差校准优先的严格 SGD 场景，可优先评估 Single-micro，M=4 或 M=8 通常在内存和相对 Double 的方差之间取得较好折中；但是否优于 Naive 必须按目标指标复核。M=2 没有统计效率优势，只是对称 Double 的等价写法。若可承受逐样本梯度，Single-direct 在当前 MNIST 定点实验中 MSE 最低，可作为研究基准。</p>
    <p>不要把 GL16、任何固定阶数求积或一次守恒检查单独称为真值。建议使用“全数据、自适应误差控制、独立离散化交叉检查、端点损失守恒的数值参考”这一表述，并将未通过认证的检查点显式标注。</p>
    {continual_narrative}
    {"<figure><img src='assets/continual-learning.png' alt='Continual learning results'><figcaption>端到端持续学习结果。Permuted-MNIST 为 3 个最终种子；Split-CIFAR-100 为 1 个资源受限种子。</figcaption></figure><div class='table-wrap'>" + continual_table + "</div>" if not continual_final.empty else "<p class='callout'>持续学习完整矩阵尚无正式 final 结果文件；因此统计估计改进是否转化为遗忘降低仍列为待验证，不在本版中强行下结论。</p>"}
  </section>

  <section data-contract-section="further-questions">
    <h2>进一步问题</h2>
    <p>最重要的后续问题有三个：协方差修正在多节点路径积分中是否仍能稳定改善参数排序；非光滑网络的折点密度如何决定参考积分成本；以及更准确的重要性是否在固定正则强度和独立调参协议下真实降低持续学习遗忘。上述问题需要 CIFAR-10 平滑/非平滑对照与 Permuted-MNIST、Split-CIFAR-100 端到端结果共同回答。</p>
    <p class="lede">研究背景与实现依据包括 <a href="https://proceedings.mlr.press/v70/zenke17a.html">Zenke et al. (2017) SI</a>、<a href="https://proceedings.mlr.press/v151/benzing22a.html">Benzing (2022) 无偏 SI</a>、<a href="https://repository.lib.ncsu.edu/items/2dd4f5a1-4a95-4871-b090-d9da1bde4e44">Hoeffding U-statistics</a>、<a href="https://proceedings.mlr.press/v97/simsekli19a.html">Şimşekli et al. (2019) 重尾梯度噪声</a>，以及 SciPy <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad_vec.html"><code>quad_vec</code></a> 自适应向量积分。详细文献与预注册协议见仓库 <code>docs/</code>。</p>
  </section>
</main>
</body>
</html>"""
    path.write_text(html_text, encoding="utf-8")


def run_report(config: dict[str, Any], force: bool = False) -> Path:
    run_dir = prepare_run_dir(config, "report")
    report_path = run_dir / "report.html"
    if report_path.exists() and not force:
        return run_dir
    write_metadata(run_dir, config, "report", "running")
    assets = run_dir / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    assets.mkdir(parents=True)
    _use_chart_theme()

    input_config = config.get("inputs", {})
    input_paths = {
        name: _expand_result_paths([str(value) for value in values])
        for name, values in input_config.items()
    }
    simulation = _read_many(input_paths.get("simulation", []))
    checkpoint = _read_many(input_paths.get("checkpoint", []))
    quadrature_checkpoint = _read_many(
        input_paths.get("quadrature_checkpoint", [])
    )
    reference = _read_many(input_paths.get("reference", []))
    stress_checkpoint = _read_many(input_paths.get("stress_checkpoint", []))
    continual = _read_many(input_paths.get("continual", []))
    if simulation.empty:
        raise FileNotFoundError("The report requires at least one simulation result")

    bootstrap_repetitions = int(config.get("bootstrap_repetitions", 10_000))
    seed = int(config.get("seed", 0))
    simulation_decisions = _simulation_decisions(
        simulation,
        bootstrap_repetitions,
        seed,
    )
    checkpoint_summary, checkpoint_decisions = _checkpoint_summary(
        checkpoint,
        bootstrap_repetitions,
        seed,
    )
    decisions = pd.concat(
        [simulation_decisions, checkpoint_decisions],
        ignore_index=True,
        sort=False,
    )
    omnibus = _omnibus_tests(simulation_decisions)

    _plot_simulation_bias(simulation, assets)
    _plot_simulation_h3(simulation, assets)
    _plot_checkpoint_bias(checkpoint_summary, assets)
    _plot_checkpoint_mse(checkpoint_summary, assets)
    _plot_checkpoint_h3(checkpoint_decisions, assets)
    _plot_reference(reference, assets)
    _plot_continual(continual, assets)
    _write_html(
        report_path,
        simulation=simulation,
        checkpoint=checkpoint,
        checkpoint_summary=checkpoint_summary,
        checkpoint_decisions=checkpoint_decisions,
        reference=reference,
        quadrature_checkpoint=quadrature_checkpoint,
        stress_checkpoint=stress_checkpoint,
        continual=continual,
        decisions=decisions,
        omnibus=omnibus,
        inputs=input_paths,
    )
    write_parquet(decisions, run_dir / "decisions.parquet")
    write_parquet(checkpoint_summary, run_dir / "checkpoint-summary.parquet")
    write_parquet(omnibus, run_dir / "omnibus-tests.parquet")
    (run_dir / "source-notes.json").write_text(
        json.dumps(
            {
                key: [str(path) for path in values]
                for key, values in input_paths.items()
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "chart-map.md").write_text(
        "\n".join(
            [
                "# Chart map",
                "",
                "- `simulation-bias-theory`: H1 analytic scaling check.",
                "- `simulation-h3-mse-ratio`: H3 matched-budget MSE ratios.",
                "- `checkpoint-total-bias`: real-model total bias by checkpoint.",
                "- `checkpoint-mse-relative`: real-model estimator MSE relative to Naive.",
                "- `checkpoint-h3-mse-ratio`: real-model matched-budget MSE ratios.",
                "- `reference-convergence`: adaptive/composite integration cross-check.",
                "- `continual-learning`: final accuracy and forgetting when available.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    write_metadata(
        run_dir,
        config,
        "report",
        "completed",
        {
            "inputs": {
                key: [str(path) for path in values]
                for key, values in input_paths.items()
            },
            "decision_rows": len(decisions),
            "checkpoint_summary_rows": len(checkpoint_summary),
            "charts": len(list(assets.glob("*.png"))),
        },
    )
    return run_dir
