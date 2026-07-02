from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dynsys_econometrics.transforms import (
    log_returns,
    pct_change,
    rolling_zscore,
    standardize_panel,
    winsorize_series,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D").tolist() * 2,
            "series_id": ["a"] * 6 + ["b"] * 6,
            "value": [100, 101, 102, 103, 104, 105, 50, 55, 60, 65, 70, 75],
        }
    )


def test_log_returns_groups_by_series() -> None:
    transformed = log_returns(_panel())
    assert transformed["series_id"].value_counts().to_dict() == {"a": 5, "b": 5}


def test_log_returns_rejects_non_positive_values() -> None:
    panel = _panel()
    panel.loc[0, "value"] = 0.0
    with pytest.raises(ValueError):
        log_returns(panel)


def test_pct_change_preserves_grouping() -> None:
    transformed = pct_change(_panel())
    assert transformed["series_id"].nunique() == 2


def test_rolling_zscore_is_trailing() -> None:
    transformed = rolling_zscore(_panel().query("series_id == 'a'"), window=3)
    assert transformed["date"].min() == pd.Timestamp("2020-01-03")


def test_standardize_panel_centered_by_group() -> None:
    transformed = standardize_panel(_panel())
    means = transformed.groupby("series_id")["value"].mean().round(8)
    assert np.allclose(means.to_numpy(), [0.0, 0.0])


def test_winsorize_series_clips_extremes() -> None:
    panel = _panel()
    panel.loc[5, "value"] = 500.0
    transformed = winsorize_series(panel, lower=0.05, upper=0.95)
    assert transformed["value"].max() < 500.0
