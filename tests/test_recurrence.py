from __future__ import annotations

import pandas as pd

from dynsys_econometrics.return_times import (
    compare_return_time_distributions,
    recurrence_rate,
    return_times,
    survival_curve,
)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=7, freq="D"),
            "series_id": ["a"] * 7,
            "exceedance": [False, True, False, False, True, False, True],
        }
    )


def test_return_times_regular_sequence() -> None:
    durations = return_times(_events())
    assert durations["duration"].tolist() == [3.0, 2.0]


def test_recurrence_rate_rolling_window() -> None:
    rates = recurrence_rate(_events(), window=3)
    assert "event_count" in rates.columns


def test_survival_curve_shape() -> None:
    durations = return_times(_events())
    curve = survival_curve(durations)
    assert {"duration", "survival_probability", "n_at_risk"}.issubset(curve.columns)


def test_compare_return_time_distributions() -> None:
    durations = return_times(_events())
    durations["regime"] = ["pre", "post"]
    summary = compare_return_time_distributions(durations, group_col="series_id", regime_col="regime")
    assert summary["n_durations"].sum() == 2
