from __future__ import annotations

import pandas as pd

from dynsys_econometrics.baselines import (
    autocorrelation_summary,
    drawdown_series,
    regime_summary,
    rolling_volatility,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=6, freq="D"),
            "series_id": ["a"] * 6,
            "value": [100.0, 105.0, 102.0, 108.0, 104.0, 110.0],
            "regime": ["pre", "pre", "pre", "post", "post", "post"],
        }
    )


def test_autocorrelation_summary_shape() -> None:
    summary = autocorrelation_summary(_panel()[["date", "series_id", "value"]], lags=[1, 2])
    assert summary.shape[0] == 2


def test_rolling_volatility_outputs_series() -> None:
    output = rolling_volatility(_panel()[["date", "series_id", "value"]], window=3)
    assert "rolling_volatility" in output.columns


def test_drawdown_series_known_path() -> None:
    output = drawdown_series(_panel()[["date", "series_id", "value"]])
    assert output["drawdown"].min() <= 0.0


def test_regime_summary_aggregates() -> None:
    summary = regime_summary(_panel(), regime_col="regime")
    assert set(summary["regime"]) == {"pre", "post"}
