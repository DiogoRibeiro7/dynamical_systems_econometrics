"""Generate article figures in a reproducible, repo-relative workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dynsys_econometrics.data import load_fred_csv, load_time_series_csv
from dynsys_econometrics.baselines import rolling_volatility
from dynsys_econometrics.econometrics import compare_event_diagnostics
from dynsys_econometrics.extremes import (
    bootstrap_threshold_sensitivity_analysis,
    bootstrap_run_length_sensitivity_analysis,
    estimate_runs_extremal_index,
    run_length_sensitivity_analysis,
    threshold_sensitivity_analysis,
)
from dynsys_econometrics.multivariate import stress_state_index
from dynsys_econometrics.plots import (
    plot_baseline_comparison,
    plot_cluster_adjusted_stress_by_run_length,
    plot_cluster_adjusted_stress_by_threshold,
    plot_clustered_vs_isolated_extremes,
    plot_conceptual_pipeline,
    plot_empirical_stress_illustration,
    plot_extremal_index_bars,
    plot_extremal_index_by_threshold,
    plot_joint_stress_timeline,
    plot_macro_financial_timeline,
    plot_multivariate_stress_heatmap,
    plot_orbit_and_observable,
    plot_return_time_distribution,
    plot_threshold_exceedances,
)
from dynsys_econometrics.return_times import compute_return_times
from dynsys_econometrics.simulation import (
    logistic_map,
    simulate_coupled_logistic_maps,
    simulate_garch11,
    simulate_iid_gaussian,
)


def _repo_root() -> Path:
    """Return the repository root from the script location."""
    return Path(__file__).resolve().parents[1]


def _build_macro_stress_series(repo_root: Path) -> pd.Series:
    """Load a macro-stress series if present, otherwise create a synthetic fallback."""
    candidate_path = repo_root / "data" / "raw" / "macro_stress_example.csv"
    if candidate_path.exists():
        panel = load_time_series_csv(candidate_path, series_id="macro_stress")
        series = panel.set_index("date")["value"].sort_index()
        series.name = "macro_stress"
        return series

    dates = pd.date_range("2000-01-01", periods=260, freq="ME")
    rng = np.random.default_rng(17)
    values = (
        np.cumsum(rng.normal(size=dates.size)) * 0.15
        + 2.0
        + 0.6 * np.sin(np.linspace(0.0, 10.0, dates.size))
    )
    series = pd.Series(values, index=dates, name="macro_stress")
    return series


def _build_stress_heatmap_frame() -> pd.DataFrame:
    """Create a synthetic multivariate stress panel for heatmap plotting."""
    coupled = simulate_coupled_logistic_maps(n_steps=600, n_series=3, coupling=0.15, seed=23)
    frame = pd.DataFrame(
        {
            "stress_a": coupled[:, 0],
            "stress_b": 0.7 * coupled[:, 1] + 0.3 * coupled[:, 2],
            "stress_c": np.abs(coupled[:, 2] - coupled[:, 0]),
        }
    )

    standardized = (frame - frame.mean()) / frame.std(ddof=0)
    rolling = standardized.abs().rolling(window=20, min_periods=1).mean()
    return rolling.iloc[::5].reset_index(drop=True)


def _build_empirical_illustration(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a small empirical macro-financial panel from cleaned FRED CSV files."""
    series_specs = [
        ("unrate", repo_root / "data" / "processed" / "fred_unrate_clean.csv", "UNRATE"),
        ("baa10ym", repo_root / "data" / "processed" / "fred_baa10ym_clean.csv", "BAA10YM"),
        ("kcfsi", repo_root / "data" / "processed" / "fred_kcfsi_clean.csv", "KCFSI"),
    ]

    series_map: dict[str, pd.Series] = {}
    for series_id, path, value_col in series_specs:
        frame = load_fred_csv(
            path,
            series_id=series_id,
            date_col="observation_date",
            value_col=value_col,
        )
        series = frame.set_index("date")["value"].sort_index().astype(float)
        series.name = series_id
        series_map[series_id] = series

    common = pd.concat(series_map.values(), axis=1, join="inner").dropna().sort_index()
    panel = pd.DataFrame({"date": common.index})
    exceedance_matrix = pd.DataFrame(index=common.index)
    summary_rows: list[dict[str, float | int | str]] = []

    for series_id in common.columns:
        values = common[series_id]
        result = estimate_runs_extremal_index(
            values.to_numpy(),
            threshold_quantile=0.90,
            run_length=3,
        )
        exceedance = values > result.threshold
        panel[series_id] = values.to_numpy()
        panel[f"{series_id}_threshold"] = result.threshold
        panel[f"{series_id}_exceedance"] = exceedance.to_numpy()
        exceedance_matrix[series_id] = exceedance.to_numpy()
        summary_rows.append(
            {
                "series_id": series_id,
                "sample_start": str(common.index.min().date()),
                "sample_end": str(common.index.max().date()),
                "n_observations": int(len(values)),
                "threshold_quantile": 0.90,
                "run_length": 3,
                "threshold": float(result.threshold),
                "n_exceedances": int(result.n_exceedances),
                "n_clusters": int(result.n_clusters),
                "theta_runs": float(result.theta_hat),
                "lambda_runs": float((result.n_exceedances / len(values)) / result.theta_hat),
            }
        )

    exceedance_frame = exceedance_matrix.copy()
    exceedance_frame.insert(0, "date", common.index)
    stress_timeline = stress_state_index(exceedance_frame)
    stress_timeline["date"] = pd.to_datetime(stress_timeline["date"], errors="raise")
    stress_timeline["joint_exceedance"] = stress_timeline["n_active"] >= 2
    panel = panel.merge(stress_timeline, on="date", how="left")

    return panel, pd.DataFrame(summary_rows)


def main() -> None:
    """Generate the article figure bundle under article/figures."""
    repo_root = _repo_root()
    output_dir = repo_root / "article" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline_stages = [
        "Raw economic series",
        "Transformations",
        "Stress region definition",
        "Exceedance events",
        "Clustering diagnostics",
        "Return-time diagnostics",
        "Econometric baselines",
        "Article evidence",
    ]
    pd.DataFrame({"stage_order": np.arange(1, len(pipeline_stages) + 1), "stage": pipeline_stages}).to_csv(
        output_dir / "01_conceptual_pipeline_source.csv",
        index=False,
    )
    plot_conceptual_pipeline(
        stages=pipeline_stages,
        output_path=output_dir / "01_conceptual_pipeline.png",
    )

    orbit = logistic_map(n_steps=5000, x0=0.17, r=4.0)
    observable = -np.log(np.abs(orbit - 0.5) + 1e-12)
    pd.DataFrame(
        {
            "step": np.arange(800, dtype=int),
            "orbit": orbit[:800],
            "observable": observable[:800],
        }
    ).to_csv(output_dir / "simulated_orbit_and_observable_source.csv", index=False)
    plot_orbit_and_observable(
        orbit=orbit[:800],
        observable=observable[:800],
        output_path=output_dir / "simulated_orbit_and_observable.png",
    )

    clustered_values = np.abs(simulate_garch11(n_steps=3000, seed=11))
    clustered_series = pd.Series(clustered_values, name="clustered_stress")
    clustered_threshold = float(np.quantile(clustered_values, 0.97))
    pd.DataFrame(
        {
            "step": np.arange(600, dtype=int),
            "value": clustered_series.iloc[:600].to_numpy(),
            "threshold": clustered_threshold,
            "exceedance": (clustered_series.iloc[:600] > clustered_threshold).to_numpy(),
        }
    ).to_csv(output_dir / "threshold_exceedances_source.csv", index=False)
    plot_threshold_exceedances(
        series=clustered_series.iloc[:600],
        threshold=clustered_threshold,
        output_path=output_dir / "threshold_exceedances.png",
        title="Threshold exceedances in clustered stress",
    )

    isolated_values = np.abs(simulate_iid_gaussian(n_steps=3000, seed=11))
    isolated_series = pd.Series(isolated_values, name="isolated_stress")
    isolated_threshold = float(np.quantile(isolated_values, 0.97))
    pd.DataFrame(
        {
            "step": np.arange(600, dtype=int),
            "clustered_value": clustered_series.iloc[:600].to_numpy(),
            "clustered_threshold": clustered_threshold,
            "clustered_exceedance": (clustered_series.iloc[:600] > clustered_threshold).to_numpy(),
            "isolated_value": isolated_series.iloc[:600].to_numpy(),
            "isolated_threshold": isolated_threshold,
            "isolated_exceedance": (isolated_series.iloc[:600] > isolated_threshold).to_numpy(),
        }
    ).to_csv(output_dir / "clustered_vs_isolated_extremes_source.csv", index=False)
    plot_clustered_vs_isolated_extremes(
        clustered_series=clustered_series.iloc[:600],
        isolated_series=isolated_series.iloc[:600],
        clustered_threshold=clustered_threshold,
        isolated_threshold=isolated_threshold,
        output_path=output_dir / "clustered_vs_isolated_extremes.png",
    )
    pd.DataFrame(
        {
            "step": np.arange(600, dtype=int),
            "clustered_value": clustered_series.iloc[:600].to_numpy(),
            "clustered_threshold": clustered_threshold,
            "clustered_exceedance": (clustered_series.iloc[:600] > clustered_threshold).to_numpy(),
            "isolated_value": isolated_series.iloc[:600].to_numpy(),
            "isolated_threshold": isolated_threshold,
            "isolated_exceedance": (isolated_series.iloc[:600] > isolated_threshold).to_numpy(),
        }
    ).to_csv(output_dir / "02_synthetic_extremes_comparison_source.csv", index=False)
    plot_clustered_vs_isolated_extremes(
        clustered_series=clustered_series.iloc[:600],
        isolated_series=isolated_series.iloc[:600],
        clustered_threshold=clustered_threshold,
        isolated_threshold=isolated_threshold,
        output_path=output_dir / "02_synthetic_extremes_comparison.png",
    )

    return_times = compute_return_times(observable, threshold=float(np.quantile(observable, 0.95)))
    pd.DataFrame({"return_time": return_times}).to_csv(
        output_dir / "return_time_distribution_source.csv",
        index=False,
    )
    plot_return_time_distribution(
        return_times=return_times,
        output_path=output_dir / "return_time_distribution.png",
    )
    pd.DataFrame({"return_time": return_times}).to_csv(
        output_dir / "03_return_time_distribution_source.csv",
        index=False,
    )
    plot_return_time_distribution(
        return_times=return_times,
        output_path=output_dir / "03_return_time_distribution.png",
    )

    sensitivity = threshold_sensitivity_analysis(
        observable,
        threshold_quantiles=[0.90, 0.93, 0.95, 0.97, 0.99],
        run_length=4,
    )
    bootstrap_sensitivity = bootstrap_threshold_sensitivity_analysis(
        observable,
        threshold_quantiles=[0.90, 0.93, 0.95, 0.97, 0.99],
        run_length=4,
        n_bootstrap=250,
        block_size=40,
        seed=29,
        ci_level=0.90,
    )
    pd.DataFrame(
        {
            "quantile": [row.quantile for row in bootstrap_sensitivity],
            "threshold": [row.threshold for row in bootstrap_sensitivity],
            "n_exceedances": [row.n_exceedances for row in bootstrap_sensitivity],
            "n_clusters": [row.n_clusters for row in bootstrap_sensitivity],
            "theta_runs": [row.theta_runs for row in bootstrap_sensitivity],
            "theta_cluster_mean": [row.theta_cluster_mean for row in bootstrap_sensitivity],
            "lambda_runs": [row.lambda_runs for row in bootstrap_sensitivity],
            "theta_runs_ci_lower": [row.theta_runs_ci_lower for row in bootstrap_sensitivity],
            "theta_runs_ci_upper": [row.theta_runs_ci_upper for row in bootstrap_sensitivity],
            "lambda_runs_ci_lower": [row.lambda_runs_ci_lower for row in bootstrap_sensitivity],
            "lambda_runs_ci_upper": [row.lambda_runs_ci_upper for row in bootstrap_sensitivity],
            "n_bootstrap": [row.n_bootstrap for row in bootstrap_sensitivity],
            "block_size": [row.block_size for row in bootstrap_sensitivity],
        }
    ).to_csv(output_dir / "extremal_index_by_threshold_source.csv", index=False)
    plot_extremal_index_by_threshold(
        quantiles=[row.quantile for row in bootstrap_sensitivity],
        theta_values=[row.theta_runs for row in bootstrap_sensitivity],
        lower_bounds=[row.theta_runs_ci_lower for row in bootstrap_sensitivity],
        upper_bounds=[row.theta_runs_ci_upper for row in bootstrap_sensitivity],
        output_path=output_dir / "extremal_index_by_threshold.png",
    )
    plot_cluster_adjusted_stress_by_threshold(
        quantiles=[row.quantile for row in bootstrap_sensitivity],
        lambda_values=[row.lambda_runs for row in bootstrap_sensitivity],
        lower_bounds=[row.lambda_runs_ci_lower for row in bootstrap_sensitivity],
        upper_bounds=[row.lambda_runs_ci_upper for row in bootstrap_sensitivity],
        output_path=output_dir / "cluster_adjusted_stress_by_threshold.png",
    )
    run_length_sensitivity = bootstrap_run_length_sensitivity_analysis(
        observable,
        threshold_quantile=0.90,
        run_lengths=[1, 2, 4, 6, 8],
        n_bootstrap=250,
        block_size=40,
        seed=31,
        ci_level=0.90,
    )
    pd.DataFrame(
        {
            "run_length": [row.run_length for row in run_length_sensitivity],
            "threshold_quantile": [row.threshold_quantile for row in run_length_sensitivity],
            "threshold": [row.threshold for row in run_length_sensitivity],
            "n_exceedances": [row.n_exceedances for row in run_length_sensitivity],
            "n_clusters": [row.n_clusters for row in run_length_sensitivity],
            "theta_runs": [row.theta_runs for row in run_length_sensitivity],
            "lambda_runs": [row.lambda_runs for row in run_length_sensitivity],
            "lambda_runs_ci_lower": [row.lambda_runs_ci_lower for row in run_length_sensitivity],
            "lambda_runs_ci_upper": [row.lambda_runs_ci_upper for row in run_length_sensitivity],
            "n_bootstrap": [row.n_bootstrap for row in run_length_sensitivity],
            "block_size": [row.block_size for row in run_length_sensitivity],
        }
    ).to_csv(output_dir / "cluster_adjusted_stress_by_run_length_source.csv", index=False)
    plot_cluster_adjusted_stress_by_run_length(
        run_lengths=[row.run_length for row in run_length_sensitivity],
        lambda_values=[row.lambda_runs for row in run_length_sensitivity],
        lower_bounds=[row.lambda_runs_ci_lower for row in run_length_sensitivity],
        upper_bounds=[row.lambda_runs_ci_upper for row in run_length_sensitivity],
        output_path=output_dir / "cluster_adjusted_stress_by_run_length.png",
    )
    extremal_index_table = pd.DataFrame(
        {
            "series_id": ["logistic_observable", "clustered_stress", "isolated_stress"],
            "extremal_index": [
                estimate_runs_extremal_index(observable, threshold_quantile=0.95, run_length=4).theta_hat,
                estimate_runs_extremal_index(clustered_values, threshold_quantile=0.95, run_length=4).theta_hat,
                estimate_runs_extremal_index(isolated_values, threshold_quantile=0.95, run_length=4).theta_hat,
            ],
        }
    )
    extremal_index_table.to_csv(output_dir / "04_extremal_index_by_series_source.csv", index=False)
    fig, _ = plot_extremal_index_bars(extremal_index_table)
    fig.tight_layout()
    fig.savefig(output_dir / "04_extremal_index_by_series.png", dpi=160)

    macro_series = _build_macro_stress_series(repo_root)
    macro_threshold = float(macro_series.quantile(0.95))
    crisis_windows = [
        (pd.Timestamp("2001-03-31"), pd.Timestamp("2002-06-30"), "dot-com"),
        (pd.Timestamp("2008-09-30"), pd.Timestamp("2010-06-30"), "gfc"),
        (pd.Timestamp("2020-03-31"), pd.Timestamp("2021-06-30"), "pandemic"),
    ]
    pd.DataFrame(
        {
            "date": macro_series.index,
            "value": macro_series.to_numpy(),
            "threshold": macro_threshold,
            "exceedance": (macro_series >= macro_threshold).to_numpy(),
        }
    ).to_csv(output_dir / "macro_financial_crisis_timeline_source.csv", index=False)
    plot_macro_financial_timeline(
        series=macro_series,
        threshold=macro_threshold,
        crisis_windows=crisis_windows,
        output_path=output_dir / "macro_financial_crisis_timeline.png",
    )

    heatmap_frame = _build_stress_heatmap_frame()
    heatmap_frame.to_csv(output_dir / "multivariate_stress_heatmap_source.csv", index=False)
    plot_multivariate_stress_heatmap(
        frame=heatmap_frame,
        output_path=output_dir / "multivariate_stress_heatmap.png",
    )
    threshold_mask = heatmap_frame.ge(heatmap_frame.quantile(0.85), axis=1).reset_index(drop=True)
    threshold_mask.insert(0, "date", pd.date_range("2000-01-31", periods=len(threshold_mask), freq="5ME"))
    stress_timeline = stress_state_index(threshold_mask)
    stress_timeline["joint_exceedance"] = stress_timeline["n_active"] >= 2
    stress_timeline.to_csv(output_dir / "05_multivariate_stress_timeline_source.csv", index=False)
    fig, _ = plot_joint_stress_timeline(stress_timeline)
    fig.tight_layout()
    fig.savefig(output_dir / "05_multivariate_stress_timeline.png", dpi=160)

    baseline_panel = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-31", periods=clustered_series.size, freq="ME"),
            "series_id": "clustered_stress",
            "value": clustered_series.to_numpy(),
        }
    )
    baseline_events = pd.DataFrame(
        {
            "date": baseline_panel["date"],
            "series_id": baseline_panel["series_id"],
            "exceedance": baseline_panel["value"] >= clustered_threshold,
        }
    )
    baseline_summary = rolling_volatility(baseline_panel, window=24)
    comparison = compare_event_diagnostics(
        baseline_events,
        baseline_summary[["date", "series_id", "rolling_volatility"]],
    )
    comparison.to_csv(output_dir / "06_econometric_vs_recurrence_diagnostics_source.csv", index=False)
    fig, _ = plot_baseline_comparison(comparison)
    fig.tight_layout()
    fig.savefig(output_dir / "06_econometric_vs_recurrence_diagnostics.png", dpi=160)

    empirical_merged, empirical_summary = _build_empirical_illustration(repo_root)
    empirical_merged.to_csv(output_dir / "07_real_data_stress_illustration_source.csv", index=False)
    empirical_summary.to_csv(output_dir / "07_real_data_stress_summary_source.csv", index=False)
    plot_empirical_stress_illustration(
        panel=empirical_merged,
        output_path=output_dir / "07_real_data_stress_illustration.png",
    )

    summary_paths = [
        output_dir / "01_conceptual_pipeline.png",
        output_dir / "02_synthetic_extremes_comparison.png",
        output_dir / "03_return_time_distribution.png",
        output_dir / "04_extremal_index_by_series.png",
        output_dir / "05_multivariate_stress_timeline.png",
        output_dir / "06_econometric_vs_recurrence_diagnostics.png",
        output_dir / "07_real_data_stress_illustration.png",
        output_dir / "simulated_orbit_and_observable.png",
        output_dir / "threshold_exceedances.png",
        output_dir / "clustered_vs_isolated_extremes.png",
        output_dir / "return_time_distribution.png",
        output_dir / "extremal_index_by_threshold.png",
        output_dir / "cluster_adjusted_stress_by_threshold.png",
        output_dir / "cluster_adjusted_stress_by_run_length.png",
        output_dir / "macro_financial_crisis_timeline.png",
        output_dir / "multivariate_stress_heatmap.png",
    ]
    for path in summary_paths:
        print(path.relative_to(repo_root))

    result = estimate_runs_extremal_index(observable, threshold_quantile=0.95, run_length=4)
    print(
        "observable_summary:",
        f"threshold={result.threshold:.4f}",
        f"theta={result.theta_hat:.4f}",
        f"clusters={result.n_clusters}",
    )


if __name__ == "__main__":
    main()
