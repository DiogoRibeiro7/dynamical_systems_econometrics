from __future__ import annotations

from pathlib import Path

import pandas as pd

from dynsys_econometrics.plots import (
    plot_baseline_comparison,
    plot_extremal_index_bars,
    plot_joint_stress_timeline,
    plot_series_with_threshold,
    plot_time_series_with_exceedances,
)


def test_plot_time_series_with_exceedances_returns_figure() -> None:
    series = pd.Series([1.0, 2.0, 4.0], index=pd.date_range("2020-01-01", periods=3, freq="D"))
    fig, ax = plot_time_series_with_exceedances(series, threshold=2.5)
    assert fig is ax.figure


def test_plot_extremal_index_bars_returns_figure() -> None:
    table = pd.DataFrame({"series_id": ["a", "b"], "extremal_index": [0.7, 0.4]})
    fig, ax = plot_extremal_index_bars(table)
    assert fig is ax.figure


def test_plot_joint_stress_timeline_returns_figure() -> None:
    table = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "stress_score": [0.1, 0.5, 0.9],
            "joint_exceedance": [False, True, True],
        }
    )
    fig, ax = plot_joint_stress_timeline(table)
    assert fig is ax.figure


def test_plot_baseline_comparison_returns_figure() -> None:
    table = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "exceedance": [False, True, False],
            "rolling_volatility": [0.1, 0.2, 0.15],
        }
    )
    fig, ax = plot_baseline_comparison(table)
    assert fig is ax.figure


def test_plot_series_with_threshold_saves_file(tmp_path: Path) -> None:
    output_path = tmp_path / "series.png"
    series = pd.Series([1.0, 2.0, 3.0], index=pd.date_range("2020-01-01", periods=3, freq="D"))
    plot_series_with_threshold(series, threshold=2.0, output_path=output_path)
    assert output_path.exists()
