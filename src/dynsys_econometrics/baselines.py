"""Baseline econometric diagnostics used for comparison."""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def _sorted_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Return a long-format panel sorted by series and date."""
    required = {"date", "series_id", "value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    return frame.sort_values(["series_id", "date"]).reset_index(drop=True)


def autocorrelation_summary(df: pd.DataFrame, lags: Sequence[int]) -> pd.DataFrame:
    """Compute per-series sample autocorrelations for requested lags."""
    if not lags:
        raise ValueError("lags must be non-empty.")
    if any(lag <= 0 for lag in lags):
        raise ValueError("lags must be strictly positive.")
    frame = _sorted_panel(df)
    rows: list[dict[str, object]] = []
    for series_id, group in frame.groupby("series_id", sort=True):
        values = group["value"]
        for lag in lags:
            rows.append(
                {
                    "series_id": str(series_id),
                    "lag": int(lag),
                    "autocorrelation": float(values.autocorr(lag=lag)),
                }
            )
    return pd.DataFrame(rows)


def rolling_volatility(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Compute trailing rolling volatility by series."""
    if window < 2:
        raise ValueError("window must be at least 2.")
    frame = _sorted_panel(df)
    output = frame.copy()
    output["rolling_volatility"] = (
        output.groupby("series_id")["value"].transform(lambda s: s.rolling(window=window, min_periods=window).std(ddof=1))
    )
    return output.dropna(subset=["rolling_volatility"]).reset_index(drop=True)


def drawdown_series(df: pd.DataFrame) -> pd.DataFrame:
    """Compute drawdown paths from level-like values by series."""
    frame = _sorted_panel(df)
    output = frame.copy()
    running_max = output.groupby("series_id")["value"].cummax()
    if (running_max <= 0).any():
        raise ValueError("drawdown_series requires strictly positive values.")
    output["drawdown"] = output["value"] / running_max - 1.0
    return output


def regime_summary(df: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    """Summarize value distributions by series and regime label."""
    required = {"series_id", "value", regime_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    summary = (
        df.groupby(["series_id", regime_col], dropna=False)["value"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    return summary.rename(columns={"count": "n_observations"})
