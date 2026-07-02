from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dynsys_econometrics.plots import (
    plot_extremal_index_by_threshold,
    plot_macro_financial_timeline,
    plot_multivariate_stress_heatmap,
    plot_orbit_and_observable,
    plot_return_time_distribution,
    plot_threshold_exceedances,
)


def test_plot_orbit_and_observable_saves_file(tmp_path: Path) -> None:
    output_path = tmp_path / "orbit.png"
    orbit = np.linspace(0.1, 0.9, 50)
    observable = -np.log(np.abs(orbit - 0.5) + 1e-6)
    plot_orbit_and_observable(orbit=orbit, observable=observable, output_path=output_path)
    assert output_path.exists()


def test_plot_threshold_exceedances_saves_file(tmp_path: Path) -> None:
    output_path = tmp_path / "threshold_exceedances.png"
    series = pd.Series(
        [1.0, 1.5, 3.0, 2.0, 4.0],
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
        name="stress",
    )
    plot_threshold_exceedances(series=series, threshold=2.5, output_path=output_path)
    assert output_path.exists()


def test_plot_return_time_distribution_saves_file(tmp_path: Path) -> None:
    output_path = tmp_path / "return_times.png"
    plot_return_time_distribution(return_times=np.array([2, 3, 3, 5], dtype=np.int64), output_path=output_path)
    assert output_path.exists()


def test_plot_extremal_index_by_threshold_saves_file(tmp_path: Path) -> None:
    output_path = tmp_path / "extremal_index.png"
    plot_extremal_index_by_threshold(
        quantiles=[0.9, 0.95, 0.99],
        theta_values=[0.8, 0.6, 0.5],
        output_path=output_path,
    )
    assert output_path.exists()


def test_plot_macro_financial_timeline_saves_file(tmp_path: Path) -> None:
    output_path = tmp_path / "timeline.png"
    index = pd.date_range("2020-01-31", periods=12, freq="ME")
    series = pd.Series(np.linspace(1.0, 3.0, 12), index=index, name="macro_stress")
    plot_macro_financial_timeline(
        series=series,
        threshold=2.2,
        crisis_windows=[(pd.Timestamp("2020-03-31"), pd.Timestamp("2020-06-30"), "shock")],
        output_path=output_path,
    )
    assert output_path.exists()


def test_plot_multivariate_stress_heatmap_saves_file(tmp_path: Path) -> None:
    output_path = tmp_path / "heatmap.png"
    frame = pd.DataFrame(
        {
            "stress_a": [0.1, 0.2, 0.3],
            "stress_b": [0.3, 0.2, 0.1],
            "stress_c": [0.2, 0.4, 0.6],
        }
    )
    plot_multivariate_stress_heatmap(frame=frame, output_path=output_path)
    assert output_path.exists()
