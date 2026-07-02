"""Generate article figures in a reproducible, repo-relative workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dynsys_econometrics.data import load_time_series_csv
from dynsys_econometrics.extremes import estimate_runs_extremal_index, threshold_sensitivity_analysis
from dynsys_econometrics.plots import (
    plot_clustered_vs_isolated_extremes,
    plot_extremal_index_by_threshold,
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


def main() -> None:
    """Generate the article figure bundle under article/figures."""
    repo_root = _repo_root()
    output_dir = repo_root / "article" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    orbit = logistic_map(n_steps=5000, x0=0.17, r=4.0)
    observable = -np.log(np.abs(orbit - 0.5) + 1e-12)
    plot_orbit_and_observable(
        orbit=orbit[:800],
        observable=observable[:800],
        output_path=output_dir / "simulated_orbit_and_observable.png",
    )

    clustered_values = np.abs(simulate_garch11(n_steps=3000, seed=11))
    clustered_series = pd.Series(clustered_values, name="clustered_stress")
    clustered_threshold = float(np.quantile(clustered_values, 0.97))
    plot_threshold_exceedances(
        series=clustered_series.iloc[:600],
        threshold=clustered_threshold,
        output_path=output_dir / "threshold_exceedances.png",
        title="Threshold exceedances in clustered stress",
    )

    isolated_values = np.abs(simulate_iid_gaussian(n_steps=3000, seed=11))
    isolated_series = pd.Series(isolated_values, name="isolated_stress")
    isolated_threshold = float(np.quantile(isolated_values, 0.97))
    plot_clustered_vs_isolated_extremes(
        clustered_series=clustered_series.iloc[:600],
        isolated_series=isolated_series.iloc[:600],
        clustered_threshold=clustered_threshold,
        isolated_threshold=isolated_threshold,
        output_path=output_dir / "clustered_vs_isolated_extremes.png",
    )

    return_times = compute_return_times(observable, threshold=float(np.quantile(observable, 0.95)))
    plot_return_time_distribution(
        return_times=return_times,
        output_path=output_dir / "return_time_distribution.png",
    )

    sensitivity = threshold_sensitivity_analysis(
        observable,
        threshold_quantiles=[0.90, 0.93, 0.95, 0.97, 0.99],
        run_length=4,
    )
    plot_extremal_index_by_threshold(
        quantiles=[row.quantile for row in sensitivity],
        theta_values=[row.theta_runs for row in sensitivity],
        output_path=output_dir / "extremal_index_by_threshold.png",
    )

    macro_series = _build_macro_stress_series(repo_root)
    macro_threshold = float(macro_series.quantile(0.95))
    crisis_windows = [
        (pd.Timestamp("2001-03-31"), pd.Timestamp("2002-06-30"), "dot-com"),
        (pd.Timestamp("2008-09-30"), pd.Timestamp("2010-06-30"), "gfc"),
        (pd.Timestamp("2020-03-31"), pd.Timestamp("2021-06-30"), "pandemic"),
    ]
    plot_macro_financial_timeline(
        series=macro_series,
        threshold=macro_threshold,
        crisis_windows=crisis_windows,
        output_path=output_dir / "macro_financial_crisis_timeline.png",
    )

    heatmap_frame = _build_stress_heatmap_frame()
    plot_multivariate_stress_heatmap(
        frame=heatmap_frame,
        output_path=output_dir / "multivariate_stress_heatmap.png",
    )

    summary_paths = [
        output_dir / "simulated_orbit_and_observable.png",
        output_dir / "threshold_exceedances.png",
        output_dir / "clustered_vs_isolated_extremes.png",
        output_dir / "return_time_distribution.png",
        output_dir / "extremal_index_by_threshold.png",
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
