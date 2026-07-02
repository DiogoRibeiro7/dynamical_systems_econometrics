"""Data-loading contracts for free macro-financial datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_time_series_csv(path: str | Path, date_col: str = "date", value_col: str = "value") -> pd.Series:
    """Load a single time series from a CSV file.

    The expected contract is one date column and one numeric value column. The function is
    deliberately strict because econometric tail analysis is sensitive to silent parsing
    errors and date misalignment.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    missing = {date_col, value_col}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame[date_col] = pd.to_datetime(frame[date_col], errors="raise")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="raise")
    series = frame.set_index(date_col)[value_col].sort_index()
    series.name = value_col
    return series
