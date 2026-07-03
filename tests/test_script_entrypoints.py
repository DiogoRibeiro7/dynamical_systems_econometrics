from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd


def _load_script_module(script_name: str):
    """Load a script module from the repository scripts directory."""
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load script module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetch_public_data_main_validates_catalog(capsys, tmp_path: Path) -> None:
    module = _load_script_module("fetch_public_data.py")
    catalog_path = tmp_path / "catalog.yaml"
    csv_path = tmp_path / "example.csv"
    pd.DataFrame({"date": ["2020-01-01"], "value": [1.0]}).to_csv(csv_path, index=False)
    catalog_path.write_text(
        "series:\n"
        "  - series_id: demo\n"
        "    description: Demo series\n"
        "    source: local_csv\n"
        f"    path: {csv_path.as_posix()}\n",
        encoding="utf-8",
    )
    original_argv = sys.argv[:]
    try:
        sys.argv = ["fetch_public_data.py", "--catalog", str(catalog_path)]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "valid=True" in captured.out


def test_fetch_public_data_main_materializes_catalog(capsys, tmp_path: Path) -> None:
    module = _load_script_module("fetch_public_data.py")
    raw_path = tmp_path / "raw.csv"
    output_path = tmp_path / "processed.csv"
    pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-02-01"],
            "value": [10.0, 20.0],
        }
    ).to_csv(raw_path, index=False)
    catalog_path = tmp_path / "catalog.yaml"
    catalog_path.write_text(
        "series:\n"
        "  - series_id: demo\n"
        "    description: Demo series\n"
        "    source: local_csv\n"
        f"    path: {raw_path.as_posix()}\n",
        encoding="utf-8",
    )
    original_argv = sys.argv[:]
    try:
        sys.argv = [
            "fetch_public_data.py",
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


def test_run_experiment_main_writes_synthetic_outputs(capsys, tmp_path: Path) -> None:
    module = _load_script_module("run_experiment.py")
    config_path = tmp_path / "synthetic.yaml"
    output_dir = tmp_path / "synthetic_outputs"
    config_path.write_text(
        "experiment:\n"
        "  name: wrapper_synth\n"
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
        sys.argv = ["run_experiment.py", "--config", str(config_path)]
        module.main()
    finally:
        sys.argv = original_argv
    captured = capsys.readouterr()
    assert "wrapper_synth" in captured.out
    assert (output_dir / "run_summary.json").exists()


def test_run_empirical_experiment_main_writes_outputs(capsys, tmp_path: Path) -> None:
    module = _load_script_module("run_empirical_experiment.py")
    csv_path = tmp_path / "empirical.csv"
    pd.DataFrame(
        {
            "date": pd.date_range("2020-01-31", periods=18, freq="ME"),
            "series_id": ["macro"] * 18,
            "value": list(range(18)),
        }
    ).to_csv(csv_path, index=False)
    config_path = tmp_path / "empirical.yaml"
    output_dir = tmp_path / "empirical_outputs"
    config_path.write_text(
        "experiment:\n"
        "  name: wrapper_empirical\n"
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
    repo_root = Path(__file__).resolve().parents[1]
    target_config = repo_root / "configs" / "empirical.yaml"
    original_content = target_config.read_text(encoding="utf-8") if target_config.exists() else None
    original_argv = sys.argv[:]
    try:
        target_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        sys.argv = ["run_empirical_experiment.py"]
        module.main()
    finally:
        sys.argv = original_argv
        if original_content is None:
            target_config.unlink(missing_ok=True)
        else:
            target_config.write_text(original_content, encoding="utf-8")
    captured = capsys.readouterr()
    assert "wrapper_empirical" in captured.out
    assert (output_dir / "run_summary.json").exists()


def test_run_empirical_experiment_main_supports_catalog_input(capsys, tmp_path: Path) -> None:
    module = _load_script_module("run_empirical_experiment.py")
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
    config_path = tmp_path / "empirical_catalog.yaml"
    output_dir = tmp_path / "empirical_catalog_outputs"
    config_path.write_text(
        "experiment:\n"
        "  name: wrapper_empirical_catalog\n"
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
    repo_root = Path(__file__).resolve().parents[1]
    target_config = repo_root / "configs" / "empirical.yaml"
    original_content = target_config.read_text(encoding="utf-8") if target_config.exists() else None
    original_argv = sys.argv[:]
    try:
        target_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        sys.argv = ["run_empirical_experiment.py"]
        module.main()
    finally:
        sys.argv = original_argv
        if original_content is None:
            target_config.unlink(missing_ok=True)
        else:
            target_config.write_text(original_content, encoding="utf-8")
    captured = capsys.readouterr()
    assert "wrapper_empirical_catalog" in captured.out
    assert materialized_path.exists()
    assert (output_dir / "run_summary.json").exists()


def test_run_synthetic_experiment_main_writes_outputs(capsys, tmp_path: Path) -> None:
    module = _load_script_module("run_synthetic_experiment.py")
    config_path = tmp_path / "synthetic.yaml"
    output_dir = tmp_path / "synthetic_outputs"
    config_path.write_text(
        "experiment:\n"
        "  name: wrapper_synthetic_direct\n"
        "  mode: synthetic\n"
        "synthetic:\n"
        "  systems:\n"
        "    - name: logistic\n"
        "      n: 90\n"
        "analysis:\n"
        "  threshold_quantile: 0.9\n"
        "  run_length: 2\n"
        "outputs:\n"
        f"  directory: {output_dir.as_posix()}\n",
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[1]
    target_config = repo_root / "configs" / "synthetic.yaml"
    original_content = target_config.read_text(encoding="utf-8") if target_config.exists() else None
    original_argv = sys.argv[:]
    try:
        target_config.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        sys.argv = ["run_synthetic_experiment.py"]
        module.main()
    finally:
        sys.argv = original_argv
        if original_content is None:
            target_config.unlink(missing_ok=True)
        else:
            target_config.write_text(original_content, encoding="utf-8")
    captured = capsys.readouterr()
    assert "wrapper_synthetic_direct" in captured.out
    assert (output_dir / "run_summary.json").exists()
