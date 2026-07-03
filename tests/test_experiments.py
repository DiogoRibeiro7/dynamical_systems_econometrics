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


def test_run_empirical_experiment_supports_loader_mapping_and_warnings(tmp_path) -> None:
    csv_path = tmp_path / "fred.csv"
    pd.DataFrame(
        {
            "DATE": pd.date_range("2020-01-31", periods=12, freq="ME"),
            "VALUE": list(range(12)),
        }
    ).to_csv(csv_path, index=False)
    config = {
        "experiment": {"name": "empirical_loader_test", "mode": "empirical"},
        "empirical": {"loader": "fred_csv", "path": str(csv_path), "series_id": "UNRATE"},
        "analysis": {"threshold_quantile": 0.8, "run_length": 2, "min_observations": 24},
    }
    result = run_empirical_experiment(config)
    assert result.tables["empirical_panel"]["series_id"].tolist() == ["UNRATE"] * 12
    assert len(result.warnings) == 1
    assert "min_observations=24" in result.warnings[0]


def test_run_empirical_experiment_supports_catalog_input(tmp_path) -> None:
    raw_path = tmp_path / "catalog_source.csv"
    materialized_path = tmp_path / "catalog_panel.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=18, freq="ME"),
            "value": list(range(1, 19)),
        }
    ).to_csv(raw_path, index=False)
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: macro\n"
        "    description: Macro proxy\n"
        "    source: local_csv\n"
        f"    path: {raw_path.as_posix()}\n",
        encoding="utf-8",
    )
    config = {
        "experiment": {"name": "empirical_catalog_test", "mode": "empirical"},
        "empirical": {"catalog_path": str(catalog_path), "output_path": str(materialized_path)},
        "analysis": {"threshold_quantile": 0.8, "run_length": 2},
    }
    result = run_empirical_experiment(config)
    assert "empirical_panel" in result.tables
    assert result.tables["empirical_panel"]["series_id"].tolist() == ["macro"] * 18


def test_run_empirical_experiment_applies_direct_transformation(tmp_path) -> None:
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=18, freq="ME"),
            "value": [100.0 + i for i in range(18)],
        }
    ).to_csv(csv_path, index=False)
    config = {
        "experiment": {"name": "empirical_transform_test", "mode": "empirical"},
        "empirical": {
            "loader": "local_csv",
            "path": str(csv_path),
            "series_id": "macro",
            "transformation": "pct_change",
        },
        "analysis": {"threshold_quantile": 0.8, "run_length": 2},
    }
    result = run_empirical_experiment(config)
    assert len(result.tables["empirical_panel"]) == 17


def test_run_empirical_experiment_supports_per_series_overrides(tmp_path) -> None:
    prices_path = tmp_path / "prices.csv"
    inflation_path = tmp_path / "inflation.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=18, freq="ME"),
            "value": [100.0 + i for i in range(18)],
        }
    ).to_csv(prices_path, index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=18, freq="ME"),
            "value": [2.0 + 0.1 * i for i in range(18)],
        }
    ).to_csv(inflation_path, index=False)
    config = {
        "experiment": {"name": "empirical_multi_series_test", "mode": "empirical"},
        "empirical": {
            "series": [
                {
                    "loader": "local_csv",
                    "path": str(prices_path),
                    "series_id": "equity",
                    "transformation": "pct_change",
                },
                {
                    "loader": "local_csv",
                    "path": str(inflation_path),
                    "series_id": "inflation",
                    "transformation": "level",
                },
            ]
        },
        "analysis": {"threshold_quantile": 0.8, "run_length": 2},
    }
    result = run_empirical_experiment(config)
    assert set(result.tables["empirical_panel"]["series_id"]) == {"equity", "inflation"}


def test_run_empirical_experiment_supports_multiple_catalogs(tmp_path) -> None:
    first_raw = tmp_path / "first.csv"
    second_raw = tmp_path / "second.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=18, freq="ME"),
            "value": [100.0 + i for i in range(18)],
        }
    ).to_csv(first_raw, index=False)
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=18, freq="ME"),
            "value": [2.0 + 0.1 * i for i in range(18)],
        }
    ).to_csv(second_raw, index=False)
    first_catalog = tmp_path / "catalog_one.yaml"
    second_catalog = tmp_path / "catalog_two.yaml"
    first_catalog.write_text(
        "series:\n"
        "  - series_id: equity\n"
        "    description: Equity levels\n"
        "    source: local_csv\n"
        f"    path: {first_raw.as_posix()}\n",
        encoding="utf-8",
    )
    second_catalog.write_text(
        "series:\n"
        "  - series_id: inflation\n"
        "    description: Inflation levels\n"
        "    source: local_csv\n"
        f"    path: {second_raw.as_posix()}\n",
        encoding="utf-8",
    )
    config = {
        "experiment": {"name": "empirical_multi_catalog_test", "mode": "empirical"},
        "empirical": {"catalog_paths": [str(first_catalog), str(second_catalog)]},
        "analysis": {"threshold_quantile": 0.8, "run_length": 2},
    }
    result = run_empirical_experiment(config)
    assert set(result.tables["empirical_panel"]["series_id"]) == {"equity", "inflation"}


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
