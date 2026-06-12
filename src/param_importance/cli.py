from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Callable

from .analyze import run_analysis
from .checkpoint import run_checkpoint_experiment
from .config import apply_overrides, load_config
from .continual import run_continual_experiment
from .reference import run_reference_experiment
from .reporting import run_report
from .simulate import run_simulation


Runner = Callable[[dict[str, Any], bool], Path]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="param-importance",
        description="Validate stochastic overestimation in path-integral parameter importance.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in [
        "simulate",
        "checkpoint",
        "reference",
        "continual",
        "analyze",
        "report",
    ]:
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True, help="YAML experiment configuration")
        child.add_argument(
            "--set",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="Override a dotted configuration key; may be repeated",
        )
        child.add_argument(
            "--force",
            action="store_true",
            help="Ignore completed-run metadata and execute again",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = apply_overrides(load_config(args.config), args.set)
    runners: dict[str, Runner] = {
        "simulate": run_simulation,
        "checkpoint": run_checkpoint_experiment,
        "reference": run_reference_experiment,
        "continual": run_continual_experiment,
        "analyze": run_analysis,
        "report": run_report,
    }
    if args.command == "checkpoint" and "seeds" in config:
        seeds = [int(value) for value in config.pop("seeds")]
        base_name = str(config.get("run_name", "checkpoint"))
        for seed in seeds:
            child = copy.deepcopy(config)
            child["seed"] = seed
            child["run_name"] = f"{base_name}-seed{seed}"
            run_dir = runners[args.command](child, args.force)
            print(run_dir.resolve())
        return 0

    run_dir = runners[args.command](config, args.force)
    print(run_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
