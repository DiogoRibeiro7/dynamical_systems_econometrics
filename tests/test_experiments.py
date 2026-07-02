from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dynsys_econometrics.experiments import (
    run_empirical_experiment,
    run_synthetic_experiment,
    save_experiment_result,
)


def test_run_synthetic_experiment_returns_registry() -> None:
    config = {
        "experiment": {"name": "synthetic_test", "mode": "synthetic"},
        "synthetic": {"systems": [{"name": "logistic", "n": 200}, {"name": "ar1", "n": 200, "phi": 0.5}]},
        "analysis": {"threshold_quantile": 0.9, "run_length": 2},
    }
    result = run_synthetic_experiment(config)
    assert "extremal_index" in result.tables
    assert result.figures["extremal_index_by_series"] == Path("figures/extremal_index_by_series.png")


def test_save_experiment_result_writes_summary(tmp_path) -> None:
    config = {
        "experiment": {"name": "synthetic_test", "mode": "synthetic"},
        "synthetic": {"systems": [{"name": "logistic", "n": 100}]},
        "analysis": {"threshold_quantile": 0.9, "run_length": 2},
    }
    result = run_synthetic_experiment(config)
    save_experiment_result(result, tmp_path)
    summary = json.loads((tmp_path / "run_summary.json").read_text(encoding="utf-8"))
    assert summary["name"] == "synthetic_test"
    assert (tmp_path / "tables" / "synthetic_panel.csv").exists()
    assert (tmp_path / "tables" / "extremal_index.csv").exists()
    assert (tmp_path / "figures" / "extremal_index_by_series.png").exists()


def test_run_empirical_experiment_uses_local_csv(tmp_path) -> None:
    csv_path = tmp_path / "empirical.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=12, freq="M"),
            "series_id": ["macro"] * 12,
            "value": range(12),
        }
    ).to_csv(csv_path, index=False)
    config = {
        "experiment": {"name": "empirical_test", "mode": "empirical"},
        "empirical": {"input_path": str(csv_path)},
        "analysis": {"threshold_quantile": 0.8, "run_length": 2},
    }
    result = run_empirical_experiment(config)
    assert "empirical_panel" in result.tables


def test_run_synthetic_experiment_rejects_invalid_system() -> None:
    config = {
        "experiment": {"name": "bad_system", "mode": "synthetic"},
        "synthetic": {"systems": [{"name": "unknown_system", "n": 100}]},
        "analysis": {"threshold_quantile": 0.9, "run_length": 2},
    }
    try:
        run_synthetic_experiment(config)
        assert False
    except ValueError as exc:
        assert "Unsupported synthetic system" in str(exc)
