"""Multivariate stress diagnostics for joint exceedance analysis."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def wide_panel(
    df: pd.DataFrame,
    date_col: str = "date",
    series_col: str = "series_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """Convert a long-format panel into a wide date-indexed matrix."""
    required = {date_col, series_col, value_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="raise")
    duplicated = frame.duplicated([date_col, series_col])
    if duplicated.any():
        raise ValueError("Duplicate date + series_id rows prevent wide conversion.")
    return frame.pivot(index=date_col, columns=series_col, values=value_col).sort_index()


def joint_exceedances(
    df: pd.DataFrame,
    thresholds: Mapping[str, float] | None = None,
    quantile: float | None = None,
    min_active: int = 2,
    date_col: str = "date",
    series_col: str = "series_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """Compute a joint stress-event table from a long-format panel."""
    if thresholds is None and quantile is None:
        raise ValueError("Provide thresholds or quantile.")
    if thresholds is not None and quantile is not None:
        raise ValueError("Provide thresholds or quantile, not both.")
    if min_active < 1:
        raise ValueError("min_active must be at least 1.")
    wide = wide_panel(df, date_col=date_col, series_col=series_col, value_col=value_col)
    if thresholds is None:
        if not 0.0 < float(quantile) < 1.0:
            raise ValueError("quantile must lie in the open interval (0, 1).")
        threshold_map = {str(column): float(wide[column].quantile(float(quantile))) for column in wide.columns}
    else:
        threshold_map = {str(key): float(value) for key, value in thresholds.items()}
    mask = pd.DataFrame(index=wide.index)
    for column in wide.columns:
        if str(column) not in threshold_map:
            raise ValueError(f"Missing threshold for series '{column}'.")
        mask[str(column)] = wide[column] >= threshold_map[str(column)]
    available = wide.notna().sum(axis=1).astype(int)
    active_counts = mask.sum(axis=1).astype(int)
    active_series = mask.apply(
        lambda row: ",".join([str(column) for column, is_active in row.items() if bool(is_active)]),
        axis=1,
    )
    missing_series = wide.isna().apply(
        lambda row: ",".join([str(column) for column, is_missing in row.items() if bool(is_missing)]),
        axis=1,
    )
    stress_score = active_counts / available.replace(0, np.nan)
    return pd.DataFrame(
        {
            "date": wide.index,
            "n_active": active_counts.to_numpy(dtype=int),
            "active_series": active_series.to_numpy(dtype=object),
            "joint_exceedance": (active_counts >= min_active).to_numpy(dtype=bool),
            "stress_score": stress_score.fillna(0.0).to_numpy(dtype=float),
            "missing_series": missing_series.to_numpy(dtype=object),
            "n_available": available.to_numpy(dtype=int),
        }
    )


def tail_dependence_coefficient(
    df: pd.DataFrame,
    x: str,
    y: str,
    quantile: float = 0.95,
) -> float:
    """Estimate an empirical upper-tail dependence probability."""
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie in the open interval (0, 1).")
    if x not in df.columns or y not in df.columns:
        if {"date", "series_id", "value"}.issubset(df.columns):
            wide = wide_panel(df)
        else:
            raise ValueError("Input must be a wide matrix or canonical long-format panel.")
    else:
        wide = df.copy()
    subset = wide[[x, y]].dropna()
    if subset.empty:
        return float("nan")
    x_threshold = float(subset[x].quantile(quantile))
    y_threshold = float(subset[y].quantile(quantile))
    conditioned = subset[x] >= x_threshold
    n_conditioned = int(conditioned.sum())
    if n_conditioned == 0:
        return float("nan")
    return float((subset.loc[conditioned, y] >= y_threshold).mean())


def stress_state_index(
    exceedance_matrix: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Aggregate a Boolean exceedance matrix into a stress-state index."""
    frame = exceedance_matrix.copy()
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="raise")
        frame = frame.drop(columns=["date"])
    else:
        dates = pd.Series(frame.index, name="date")
    if frame.empty:
        raise ValueError("exceedance_matrix must not be empty.")
    frame = frame.astype(bool)
    if weights is None:
        weight_map = {str(column): 1.0 for column in frame.columns}
    else:
        weight_map = {str(column): float(weights.get(str(column), 1.0)) for column in frame.columns}
    weight_series = pd.Series(weight_map)
    numeric = frame.astype(float)
    raw_score = numeric.mul(weight_series, axis=1).sum(axis=1)
    max_score = float(weight_series.sum())
    active_series = frame.apply(
        lambda row: ",".join([str(column) for column, is_active in row.items() if bool(is_active)]),
        axis=1,
    )
    return pd.DataFrame(
        {
            "date": dates.to_numpy(),
            "n_active": frame.sum(axis=1).to_numpy(dtype=int),
            "active_series": active_series.to_numpy(dtype=object),
            "stress_score": (raw_score / max_score).to_numpy(dtype=float),
        }
    )
