from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import canonical_config, config_hash


def git_commit(cwd: str | Path | None = None) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def prepare_run_dir(
    config: dict[str, Any],
    command: str,
    output_root: str | Path | None = None,
) -> Path:
    root = Path(output_root or config.get("output_root", "outputs"))
    run_name = str(config.get("run_name", command))
    run_dir = root / command / f"{run_name}-{config_hash(config)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        delete=False,
        suffix=".tmp",
    ) as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def write_metadata(
    run_dir: str | Path,
    config: dict[str, Any],
    command: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "command": command,
        "config_hash": config_hash(config),
        "git_commit": git_commit(),
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
        },
        "config": canonical_config(config),
    }
    if extra:
        payload.update(extra)
    atomic_write_json(Path(run_dir) / "metadata.json", payload)


def write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)


def is_completed(run_dir: str | Path, required: tuple[str, ...] = ("results.parquet",)) -> bool:
    directory = Path(run_dir)
    metadata_path = directory / "metadata.json"
    if not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    revision_matches = metadata.get("git_commit") in {git_commit(), "unknown"}
    return metadata.get("status") == "completed" and revision_matches and all(
        (directory / name).exists() for name in required
    )


def discover_results(paths: list[str | Path]) -> list[Path]:
    discovered: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix == ".parquet":
            discovered.append(path)
        elif path.is_dir():
            discovered.extend(sorted(path.rglob("results.parquet")))
    return sorted(set(discovered))
