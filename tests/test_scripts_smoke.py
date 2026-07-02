from __future__ import annotations

import importlib.util
from pathlib import Path


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


def test_run_smoke_experiment_main_prints_summary(capsys) -> None:
    module = _load_script_module("run_smoke_experiment.py")
    module.main()
    captured = capsys.readouterr()
    assert "logistic_map:" in captured.out
    assert "ar1_abs:" in captured.out


def test_generate_article_figures_writes_expected_bundle(tmp_path, monkeypatch) -> None:
    module = _load_script_module("generate_article_figures.py")
    monkeypatch.setattr(module, "_repo_root", lambda: tmp_path)
    (tmp_path / "article").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "raw").mkdir(parents=True, exist_ok=True)

    module.main()

    output_dir = tmp_path / "article" / "figures"
    expected_files = {
        "01_conceptual_pipeline.png",
        "01_conceptual_pipeline_source.csv",
        "02_synthetic_extremes_comparison.png",
        "02_synthetic_extremes_comparison_source.csv",
        "03_return_time_distribution.png",
        "03_return_time_distribution_source.csv",
        "04_extremal_index_by_series.png",
        "04_extremal_index_by_series_source.csv",
        "05_multivariate_stress_timeline.png",
        "05_multivariate_stress_timeline_source.csv",
        "06_econometric_vs_recurrence_diagnostics.png",
        "06_econometric_vs_recurrence_diagnostics_source.csv",
        "simulated_orbit_and_observable.png",
        "simulated_orbit_and_observable_source.csv",
        "threshold_exceedances.png",
        "threshold_exceedances_source.csv",
        "clustered_vs_isolated_extremes.png",
        "clustered_vs_isolated_extremes_source.csv",
        "return_time_distribution.png",
        "return_time_distribution_source.csv",
        "extremal_index_by_threshold.png",
        "extremal_index_by_threshold_source.csv",
        "macro_financial_crisis_timeline.png",
        "macro_financial_crisis_timeline_source.csv",
        "multivariate_stress_heatmap.png",
        "multivariate_stress_heatmap_source.csv",
    }
    assert output_dir.exists()
    assert expected_files.issubset({path.name for path in output_dir.iterdir()})
