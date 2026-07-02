"""Econometric transformations used before rare-event analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a strictly positive price series."""
    if not isinstance(prices, pd.Series):
        raise TypeError("prices must be a pandas Series.")
    if (prices <= 0).any():
        raise ValueError("prices must be strictly positive to compute log returns.")

    return np.log(prices).diff().dropna()


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling z-score for crisis-threshold construction."""
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series.")
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer greater than 1.")

    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std(ddof=1)
    return ((series - rolling_mean) / rolling_std).dropna()
