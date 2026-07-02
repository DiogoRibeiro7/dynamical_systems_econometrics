"""Econometric transformations used before rare-event analysis."""

from __future__ import annotations

import pandas as pd

from dynsys_econometrics.transforms import log_returns as _log_returns
from dynsys_econometrics.transforms import rolling_zscore as _rolling_zscore


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a strictly positive price series."""
    return _log_returns(prices)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Compute a rolling z-score for crisis-threshold construction."""
    return _rolling_zscore(series, window)


def compare_event_diagnostics(
    event_table: pd.DataFrame,
    baseline_table: pd.DataFrame,
    on: str | list[str] = "date",
) -> pd.DataFrame:
    """Align event diagnostics with baseline econometric indicators.

    Parameters
    ----------
    event_table:
        Rare-event or recurrence output table.
    baseline_table:
        Econometric baseline table such as rolling volatility or drawdown.
    on:
        Merge key or keys. ``series_id`` is added automatically when present in
        both tables and not already listed.

    Returns
    -------
    pd.DataFrame
        Inner-joined comparison table.

    Assumptions
    -----------
    The returned merge is descriptive and alignment-based. It does not imply
    causal structure.

    Raises
    ------
    ValueError
        If the merge keys are missing.
    """
    keys = [on] if isinstance(on, str) else list(on)
    if "series_id" in event_table.columns and "series_id" in baseline_table.columns and "series_id" not in keys:
        keys.append("series_id")
    missing_event = [key for key in keys if key not in event_table.columns]
    missing_baseline = [key for key in keys if key not in baseline_table.columns]
    if missing_event or missing_baseline:
        raise ValueError(
            f"Missing merge keys. event_table missing {missing_event}; baseline_table missing {missing_baseline}."
        )
    return event_table.merge(baseline_table, on=keys, how="inner")
