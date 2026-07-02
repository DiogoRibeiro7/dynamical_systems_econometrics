from __future__ import annotations

import pandas as pd
import pytest

from dynsys_econometrics.contracts import TimeSeriesFrame
from dynsys_econometrics.extremes import mean_excess_table, threshold_exceedances
from dynsys_econometrics.multivariate import joint_exceedances
from dynsys_econometrics.return_times import survival_curve
from dynsys_econometrics.transforms import standardize_panel


def test_timeseries_frame_allow_missing_accepts_nan_but_not_inf() -> None:
    frame = TimeSeriesFrame(
        pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "series_id": ["a", "a"],
                "value": [1.0, None],
            }
        ),
        allow_missing=True,
    )
    assert frame.to_frame()["value"].isna().sum() == 1

    with pytest.raises(ValueError):
        TimeSeriesFrame(
            pd.DataFrame(
                {
                    "date": ["2020-01-01", "2020-01-02"],
                    "series_id": ["a", "a"],
                    "value": [1.0, float("inf")],
                }
            ),
            allow_missing=True,
        )


def test_threshold_exceedances_rejects_ambiguous_threshold_specification() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "series_id": ["a", "a", "a"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError):
        threshold_exceedances(panel, threshold=2.0, quantile=0.9)


def test_mean_excess_table_rejects_empty_quantiles() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "series_id": ["a", "a", "a"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError):
        mean_excess_table(panel, quantiles=[])


def test_survival_curve_rejects_non_positive_durations() -> None:
    durations = pd.DataFrame(
        {
            "series_id": ["a", "a"],
            "duration": [0.0, 2.0],
        }
    )
    with pytest.raises(ValueError):
        survival_curve(durations)


def test_standardize_panel_rejects_zero_variance_series() -> None:
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3, freq="D"),
            "series_id": ["a", "a", "a"],
            "value": [2.0, 2.0, 2.0],
        }
    )
    with pytest.raises(ValueError):
        standardize_panel(panel)


def test_joint_exceedances_rejects_missing_threshold_mapping() -> None:
    panel = pd.DataFrame(
        {
            "date": list(pd.date_range("2020-01-01", periods=3, freq="D")) * 2,
            "series_id": ["a"] * 3 + ["b"] * 3,
            "value": [1.0, 2.0, 3.0, 0.5, 1.5, 2.5],
        }
    )
    with pytest.raises(ValueError):
        joint_exceedances(panel, thresholds={"a": 2.0})
