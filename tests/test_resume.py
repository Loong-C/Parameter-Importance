from pathlib import Path

from param_importance.simulate import run_simulation


def test_completed_simulation_is_reused(tmp_path: Path) -> None:
    config = {
        "run_name": "resume",
        "output_root": str(tmp_path),
        "seed": 0,
        "repetitions": 10,
        "batch_sizes": [8],
        "microbatches": [2],
        "populations": [{"name": "gaussian"}],
    }
    first = run_simulation(config)
    result = first / "results.parquet"
    modified = result.stat().st_mtime_ns
    second = run_simulation(config)
    assert first == second
    assert result.stat().st_mtime_ns == modified

