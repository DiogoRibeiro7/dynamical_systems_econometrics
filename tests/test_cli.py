from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pandas as pd


def test_package_cli_help_prints_usage(capsys) -> None:
    module = importlib.import_module("dynsys_econometrics.__main__")
    original_argv = sys.argv[:]
    try:
        sys.argv = ["python -m dynsys_econometrics"]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


def test_package_cli_smoke_runs(capsys) -> None:
    module = importlib.import_module("dynsys_econometrics.__main__")
    original_argv = sys.argv[:]
    try:
        sys.argv = ["python -m dynsys_econometrics", "smoke"]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "logistic_map:" in captured.out


def test_package_cli_validate_catalog_runs(capsys, tmp_path: Path) -> None:
    module = importlib.import_module("dynsys_econometrics.__main__")
    catalog_path = tmp_path / "catalog.yaml"
    csv_path = tmp_path / "example.csv"
    pd.DataFrame({"date": ["2020-01-01"], "value": [1.0]}).to_csv(csv_path, index=False)
    catalog_path.write_text(
        "series:\n"
        "  - series_id: example_series\n"
        "    description: Example series\n"
        "    source: local_csv\n"
        f"    path: {csv_path.as_posix()}\n",
        encoding="utf-8",
    )
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "python -m dynsys_econometrics",
            "validate-catalog",
            "--catalog",
            str(catalog_path),
        ]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "valid=True" in captured.out


def test_package_cli_materialize_catalog_writes_output(capsys, tmp_path: Path) -> None:
    module = importlib.import_module("dynsys_econometrics.__main__")
    csv_path = tmp_path / "example.csv"
    output_path = tmp_path / "catalog_panel.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01"],
            "value": [1.0, 2.0],
        }
    ).to_csv(csv_path, index=False)
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: example_series\n"
        "    description: Example series\n"
        "    source: local_csv\n"
        f"    path: {csv_path.as_posix()}\n",
        encoding="utf-8",
    )
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "python -m dynsys_econometrics",
            "materialize-catalog",
            "--catalog",
            str(catalog_path),
            "--output",
            str(output_path),
        ]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "n_rows=2" in captured.out
    assert output_path.exists()


def test_package_cli_synthetic_writes_outputs(capsys, tmp_path: Path) -> None:
    module = importlib.import_module("dynsys_econometrics.__main__")
    config_path = tmp_path / "synthetic.yaml"
    output_dir = tmp_path / "synthetic_outputs"
    config_path.write_text(
        "experiment:\n"
        "  name: cli_synthetic_test\n"
        "  mode: synthetic\n"
        "synthetic:\n"
        "  systems:\n"
        "    - name: logistic\n"
        "      n: 120\n"
        "analysis:\n"
        "  threshold_quantile: 0.9\n"
        "  run_length: 2\n"
        "outputs:\n"
        f"  directory: {output_dir.as_posix()}\n",
        encoding="utf-8",
    )
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "python -m dynsys_econometrics",
            "synthetic",
            "--config",
            str(config_path),
        ]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "cli_synthetic_test" in captured.out
    assert (output_dir / "run_summary.json").exists()


def test_package_cli_empirical_writes_outputs(capsys, tmp_path: Path) -> None:
    module = importlib.import_module("dynsys_econometrics.__main__")
    csv_path = tmp_path / "empirical.csv"
    output_dir = tmp_path / "empirical_outputs"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=24, freq="ME"),
            "series_id": ["macro"] * 24,
            "value": list(range(24)),
        }
    ).to_csv(csv_path, index=False)
    config_path = tmp_path / "empirical.yaml"
    config_path.write_text(
        "experiment:\n"
        "  name: cli_empirical_test\n"
        "  mode: empirical\n"
        "empirical:\n"
        f"  input_path: {csv_path.as_posix()}\n"
        "analysis:\n"
        "  threshold_quantile: 0.9\n"
        "  run_length: 2\n"
        "outputs:\n"
        f"  directory: {output_dir.as_posix()}\n",
        encoding="utf-8",
    )
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "python -m dynsys_econometrics",
            "empirical",
            "--config",
            str(config_path),
        ]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "cli_empirical_test" in captured.out
    assert (output_dir / "run_summary.json").exists()


def test_package_cli_empirical_accepts_loader_config(capsys, tmp_path: Path) -> None:
    module = importlib.import_module("dynsys_econometrics.__main__")
    csv_path = tmp_path / "fred.csv"
    output_dir = tmp_path / "empirical_loader_outputs"
    pd.DataFrame(
        {
            "DATE": pd.date_range("2020-01-31", periods=24, freq="ME"),
            "VALUE": list(range(24)),
        }
    ).to_csv(csv_path, index=False)
    config_path = tmp_path / "empirical_loader.yaml"
    config_path.write_text(
        "experiment:\n"
        "  name: cli_empirical_loader_test\n"
        "  mode: empirical\n"
        "empirical:\n"
        "  loader: fred_csv\n"
        f"  path: {csv_path.as_posix()}\n"
        "  series_id: UNRATE\n"
        "analysis:\n"
        "  threshold_quantile: 0.9\n"
        "  run_length: 2\n"
        "  min_observations: 12\n"
        "outputs:\n"
        f"  directory: {output_dir.as_posix()}\n",
        encoding="utf-8",
    )
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "python -m dynsys_econometrics",
            "empirical",
            "--config",
            str(config_path),
        ]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "cli_empirical_loader_test" in captured.out
    assert (output_dir / "run_summary.json").exists()


def test_package_cli_empirical_accepts_catalog_config(capsys, tmp_path: Path) -> None:
    module = importlib.import_module("dynsys_econometrics.__main__")
    raw_path = tmp_path / "catalog_source.csv"
    materialized_path = tmp_path / "catalog_panel.csv"
    output_dir = tmp_path / "empirical_catalog_outputs"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=24, freq="ME"),
            "value": list(range(1, 25)),
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
    config_path = tmp_path / "empirical_catalog.yaml"
    config_path.write_text(
        "experiment:\n"
        "  name: cli_empirical_catalog_test\n"
        "  mode: empirical\n"
        "empirical:\n"
        f"  catalog_path: {catalog_path.as_posix()}\n"
        f"  output_path: {materialized_path.as_posix()}\n"
        "analysis:\n"
        "  threshold_quantile: 0.9\n"
        "  run_length: 2\n"
        "outputs:\n"
        f"  directory: {output_dir.as_posix()}\n",
        encoding="utf-8",
    )
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "python -m dynsys_econometrics",
            "empirical",
            "--config",
            str(config_path),
        ]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "cli_empirical_catalog_test" in captured.out
    assert materialized_path.exists()
    assert (output_dir / "run_summary.json").exists()
