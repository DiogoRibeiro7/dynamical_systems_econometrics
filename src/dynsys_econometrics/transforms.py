"""Panel transformations for empirical macro-financial series."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _sorted_panel(df: pd.DataFrame, group_col: str = "series_id") -> pd.DataFrame:
    """Return a deterministic long-format panel sorted by series and date."""
    required = {group_col, "date", "value"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["value"] = pd.to_numeric(frame["value"], errors="raise")
    return frame.sort_values([group_col, "date"]).reset_index(drop=True)


def log_returns(df: pd.DataFrame | pd.Series, value_col: str = "value") -> pd.DataFrame | pd.Series:
    """Compute grouped log returns without look-ahead.

    Parameters
    ----------
    df:
        Long-format panel or a single price series.
    value_col:
        Value column when a DataFrame is supplied.

    Returns
    -------
    pd.DataFrame | pd.Series
        Log-return output with rows lost only through first differencing.

    Raises
    ------
    ValueError
        If non-positive values appear.
    """
    if isinstance(df, pd.Series):
        if (df <= 0).any():
            raise ValueError("prices must be strictly positive to compute log returns.")
        return np.log(df).diff().dropna()
    frame = _sorted_panel(df)
    if (frame[value_col] <= 0).any():
        raise ValueError("All values must be strictly positive for log returns.")
    transformed = frame.copy()
    transformed[value_col] = transformed.groupby("series_id")[value_col].transform(lambda s: np.log(s).diff())
    return transformed.dropna(subset=[value_col]).reset_index(drop=True)


def pct_change(df: pd.DataFrame, value_col: str = "value") -> pd.DataFrame:
    """Compute grouped percentage changes in long format."""
    frame = _sorted_panel(df)
    transformed = frame.copy()
    transformed[value_col] = transformed.groupby("series_id")[value_col].pct_change()
    return transformed.dropna(subset=[value_col]).reset_index(drop=True)


def rolling_zscore(df: pd.DataFrame | pd.Series, window: int, value_col: str = "value") -> pd.DataFrame | pd.Series:
    """Compute a trailing rolling z-score by series.

    Parameters
    ----------
    df:
        Long-format panel or a numeric Series.
    window:
        Trailing window length. Must exceed one.
    value_col:
        Value column for DataFrame input.

    Returns
    -------
    pd.DataFrame | pd.Series
        Z-scored values with rows dropped where the rolling window is not yet
        available.
    """
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer greater than 1.")
    if isinstance(df, pd.Series):
        rolling_mean = df.rolling(window=window, min_periods=window).mean()
        rolling_std = df.rolling(window=window, min_periods=window).std(ddof=1)
        return ((df - rolling_mean) / rolling_std).dropna()
    frame = _sorted_panel(df)
    transformed = frame.copy()
    grouped = transformed.groupby("series_id")[value_col]
    mean = grouped.transform(lambda s: s.rolling(window=window, min_periods=window).mean())
    std = grouped.transform(lambda s: s.rolling(window=window, min_periods=window).std(ddof=1))
    transformed[value_col] = (transformed[value_col] - mean) / std
    return transformed.dropna(subset=[value_col]).reset_index(drop=True)


def standardize_panel(df: pd.DataFrame, group_col: str = "series_id") -> pd.DataFrame:
    """Standardize values within each series of a long-format panel."""
    frame = _sorted_panel(df, group_col=group_col)
    transformed = frame.copy()
    grouped = transformed.groupby(group_col)["value"]
    mean = grouped.transform("mean")
    std = grouped.transform(lambda s: s.std(ddof=0))
    if (std == 0).any():
        raise ValueError("Standard deviation must be non-zero within each series.")
    transformed["value"] = (transformed["value"] - mean) / std
    return transformed


def winsorize_series(df: pd.DataFrame, lower: float, upper: float) -> pd.DataFrame:
    """Winsorize values within each series using grouped quantiles."""
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("Require 0 <= lower < upper <= 1.")
    frame = _sorted_panel(df)
    transformed = frame.copy()
    lower_bounds = transformed.groupby("series_id")["value"].transform(lambda s: s.quantile(lower))
    upper_bounds = transformed.groupby("series_id")["value"].transform(lambda s: s.quantile(upper))
    transformed["value"] = transformed["value"].clip(lower=lower_bounds, upper=upper_bounds)
    return transformed
