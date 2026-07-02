"""Hitting-time and return-time utilities for rare-event regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import expon, kstest

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class ReturnTimeDistribution:
    """Observed return-time distribution statistics."""

    return_times: IntArray
    unique_times: IntArray
    probabilities: FloatArray
    cdf: FloatArray
    mean: float
    median: float


@dataclass(frozen=True)
class ExponentialComparisonResult:
    """Diagnostic versus exponential waiting-time benchmark."""

    lambda_hat: float
    ks_statistic: float
    ks_pvalue: float
    n: int
    empirical_mean: float


def _coerce_numeric_series(values: Sequence[float] | FloatArray) -> FloatArray:
    """Validate and convert input to finite float array."""
    if isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=np.float64)
    elif isinstance(values, (list, tuple)):
        arr = np.asarray(values, dtype=np.float64)
    else:
        try:
            import pandas as pd

            if isinstance(values, pd.Series):
                arr = values.to_numpy(dtype=np.float64)
            else:
                raise TypeError
        except Exception as exc:  # pragma: no cover
            raise TypeError(
                "values must be a one-dimensional np.ndarray, list/tuple, or pd.Series."
            ) from exc

    if arr.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")
    if arr.size == 0:
        raise ValueError("values must not be empty.")
    if not np.isfinite(arr).all():
        raise ValueError("values must contain finite values only.")
    return arr.astype(np.float64, copy=False)


def first_hitting_time(values: FloatArray | Sequence[float], threshold: float, start_index: int = 0) -> int:
    """Return the first index where values exceed threshold.

    Returns `-1` when no hitting event is found. This is intentional and not treated as
    an error.
    """
    values_arr = _coerce_numeric_series(values)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a non-negative integer.")
    if start_index >= values_arr.size:
        return -1

    positions = np.flatnonzero(values_arr > threshold)
    if positions.size == 0:
        return -1
    qualifying = positions[positions >= start_index]
    if qualifying.size == 0:
        return -1
    return int(qualifying[0])


def compute_return_times(values: FloatArray | Sequence[float], threshold: float) -> IntArray:
    """Compute gaps between consecutive threshold exceedances.

    In econometric applications, these gaps represent the recurrence time between crisis
    episodes, stress periods, volatility bursts, or other rare-event regions.
    """
    values_arr = _coerce_numeric_series(values)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    positions = np.flatnonzero(values_arr > threshold)
    if positions.size < 2:
        return np.array([], dtype=np.int64)

    return np.diff(positions).astype(np.int64)


def return_time_distribution(
    values: FloatArray | Sequence[float],
    threshold: float,
) -> ReturnTimeDistribution:
    """Compute the empirical return-time distribution from a series and threshold."""
    return_times = compute_return_times(values, threshold=threshold)
    if return_times.size == 0:
        return ReturnTimeDistribution(
            return_times=return_times,
            unique_times=np.array([], dtype=np.int64),
            probabilities=np.array([], dtype=np.float64),
            cdf=np.array([], dtype=np.float64),
            mean=float("nan"),
            median=float("nan"),
        )

    unique_times, counts = np.unique(return_times, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    cdf = np.cumsum(probs)
    return ReturnTimeDistribution(
        return_times=return_times,
        unique_times=unique_times.astype(np.int64),
        probabilities=probs,
        cdf=cdf,
        mean=float(np.mean(return_times)),
        median=float(np.median(return_times)),
    )


def empirical_survival_curve(
    return_times: FloatArray | IntArray | Sequence[float] | Sequence[int],
) -> tuple[IntArray, FloatArray]:
    """Compute empirical survival curve S(t)=P(T>=t) from return times."""
    if isinstance(return_times, np.ndarray):
        times = np.asarray(return_times, dtype=np.int64)
    else:
        times = np.asarray(list(return_times), dtype=np.int64)

    if times.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
        )
    if np.any(times < 1):
        raise ValueError("return_times must be positive integers.")

    n = int(times.size)
    unique_times, counts = np.unique(times, return_counts=True)
    tail_counts = np.flip(np.cumsum(np.flip(counts.astype(np.float64))))
    survival = tail_counts / float(n)
    return unique_times.astype(np.int64), survival.astype(np.float64)


def exponential_benchmark_comparison(
    return_times: FloatArray | IntArray | Sequence[float] | Sequence[int],
) -> ExponentialComparisonResult:
    """Fit exponential benchmark and compute a KS-style comparison.

    If return times are too short to estimate robustly, this still returns a best-effort
    benchmark fit, but `ks_pvalue` may be large in small samples.
    """
    if isinstance(return_times, np.ndarray):
        times = np.asarray(return_times, dtype=np.float64)
    else:
        times = np.asarray(list(return_times), dtype=np.float64)

    if times.size == 0:
        raise ValueError("return_times must not be empty.")
    if np.any(times < 1.0) or not np.isfinite(times).all():
        raise ValueError("return_times must be finite positive values.")

    empirical_mean = float(np.mean(times))
    lambda_hat = 1.0 / empirical_mean

    ks_result = kstest(times, lambda x: expon.cdf(x, scale=1.0 / lambda_hat))
    return ExponentialComparisonResult(
        lambda_hat=float(lambda_hat),
        ks_statistic=float(ks_result.statistic),
        ks_pvalue=float(ks_result.pvalue),
        n=int(times.size),
        empirical_mean=empirical_mean,
    )


def ks_exponential_diagnostic(
    values: FloatArray | Sequence[float],
    threshold: float,
    alpha: float = 0.05,
) -> tuple[float, float, bool, int]:
    """KS-style diagnostic directly from a sample against exponential benchmark.

    Returns:
        ks_statistic, ks_pvalue, reject_null, n_samples
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")

    benchmark = exponential_benchmark_comparison(compute_return_times(values, threshold))
    return (
        benchmark.ks_statistic,
        benchmark.ks_pvalue,
        benchmark.ks_pvalue < alpha,
        benchmark.n,
    )


def _validate_event_table(
    events: pd.DataFrame,
    *,
    date_col: str,
    event_col: str,
    group_col: str,
) -> pd.DataFrame:
    """Validate a long-format event table for recurrence diagnostics."""
    required = {date_col, event_col, group_col}
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    frame = events.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="ignore")
    frame[event_col] = frame[event_col].astype(bool)
    frame = frame.sort_values([group_col, date_col]).reset_index(drop=True)
    return frame


def inter_event_durations(
    events: pd.DataFrame,
    date_col: str = "date",
    event_col: str = "exceedance",
    group_col: str = "series_id",
) -> pd.DataFrame:
    """Compute durations between consecutive event dates.

    Parameters
    ----------
    events:
        Event table containing one row per observation and a boolean event flag.
    date_col:
        Date or step-like column.
    event_col:
        Boolean event indicator column.
    group_col:
        Series identifier column.

    Returns
    -------
    pd.DataFrame
        Duration table with ``series_id``, ``event_date``,
        ``previous_event_date``, ``duration``, and ``unit``.

    Assumptions
    -----------
    The function computes event-level durations unless the input has already
    been aggregated to one row per cluster.

    Raises
    ------
    ValueError
        If the required event table columns are missing.
    """
    frame = _validate_event_table(events, date_col=date_col, event_col=event_col, group_col=group_col)
    rows: list[dict[str, object]] = []
    for series_id, group in frame.groupby(group_col, sort=True):
        active = group[group[event_col]].copy()
        if len(active) < 2:
            continue
        dates = active[date_col]
        if pd.api.types.is_datetime64_any_dtype(dates):
            deltas = dates.diff().dt.days
            unit = "days"
        else:
            deltas = pd.to_numeric(dates, errors="raise").diff()
            unit = "steps"
        for idx in range(1, len(active)):
            rows.append(
                {
                    group_col: str(series_id),
                    "event_date": active.iloc[idx][date_col],
                    "previous_event_date": active.iloc[idx - 1][date_col],
                    "duration": float(deltas.iloc[idx]),
                    "unit": unit,
                }
            )
    return pd.DataFrame(rows)


def return_times(
    events: pd.DataFrame,
    date_col: str = "date",
    event_col: str = "exceedance",
    group_col: str = "series_id",
) -> pd.DataFrame:
    """Return inter-event durations in tidy long format.

    Parameters
    ----------
    events:
        Event table containing a boolean event indicator.
    date_col:
        Event date column.
    event_col:
        Event indicator column.
    group_col:
        Series identifier column.

    Returns
    -------
    pd.DataFrame
        Tidy duration table.

    Assumptions
    -----------
    The input is event-level data unless the caller explicitly aggregates to
    clusters first.

    Raises
    ------
    ValueError
        If the event table is malformed.
    """
    return inter_event_durations(events, date_col=date_col, event_col=event_col, group_col=group_col)


def recurrence_rate(
    events: pd.DataFrame,
    window: int | str,
    date_col: str = "date",
    event_col: str = "exceedance",
    group_col: str = "series_id",
) -> pd.DataFrame:
    """Compute rolling or resampled recurrence counts by series.

    Parameters
    ----------
    events:
        Event table with a boolean event flag.
    window:
        Integer observation window or a pandas offset string such as ``"90D"``.
    date_col:
        Date column.
    event_col:
        Event indicator column.
    group_col:
        Series identifier column.

    Returns
    -------
    pd.DataFrame
        Table with event counts by date and series.

    Raises
    ------
    ValueError
        If ``window`` is invalid for the supplied date type.
    """
    frame = _validate_event_table(events, date_col=date_col, event_col=event_col, group_col=group_col)
    outputs: list[pd.DataFrame] = []
    for series_id, group in frame.groupby(group_col, sort=True):
        local = group.copy()
        if isinstance(window, int):
            if window <= 0:
                raise ValueError("window must be positive.")
            local["event_count"] = local[event_col].astype(int).rolling(window=window, min_periods=1).sum()
            local["window"] = str(window)
            local["unit"] = "observations"
            outputs.append(local[[group_col, date_col, "event_count", "window", "unit"]])
        else:
            local[date_col] = pd.to_datetime(local[date_col], errors="raise")
            resampled = (
                local.set_index(date_col)[event_col]
                .resample(window)
                .sum()
                .reset_index(name="event_count")
            )
            resampled[group_col] = str(series_id)
            resampled["window"] = window
            resampled["unit"] = "calendar"
            outputs.append(resampled[[group_col, date_col, "event_count", "window", "unit"]])
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame(
        columns=[group_col, date_col, "event_count", "window", "unit"]
    )


def survival_curve(
    durations: pd.DataFrame,
    duration_col: str = "duration",
    group_col: str = "series_id",
) -> pd.DataFrame:
    """Estimate an empirical survival curve for return-time durations.

    Parameters
    ----------
    durations:
        Duration table containing at least ``series_id`` and ``duration``.
    duration_col:
        Positive duration column.
    group_col:
        Series identifier column.

    Returns
    -------
    pd.DataFrame
        Survival-curve table with ``duration``, ``survival_probability``, and
        ``n_at_risk`` by series.

    Raises
    ------
    ValueError
        If required columns are missing or durations are invalid.
    """
    required = {group_col, duration_col}
    missing = required.difference(durations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for series_id, group in durations.groupby(group_col, sort=True):
        values = pd.to_numeric(group[duration_col], errors="raise").to_numpy(dtype=np.float64)
        if values.size == 0:
            continue
        if np.any(values <= 0.0):
            raise ValueError("duration values must be strictly positive.")
        unique_values, counts = np.unique(values, return_counts=True)
        at_risk = np.flip(np.cumsum(np.flip(counts.astype(np.int64))))
        survival = at_risk / at_risk[0]
        for duration_value, n_risk, probability in zip(unique_values, at_risk, survival, strict=True):
            rows.append(
                {
                    group_col: str(series_id),
                    "duration": float(duration_value),
                    "survival_probability": float(probability),
                    "n_at_risk": int(n_risk),
                }
            )
    return pd.DataFrame(rows)


def compare_return_time_distributions(
    durations: pd.DataFrame,
    group_col: str,
    regime_col: str,
) -> pd.DataFrame:
    """Summarize return-time distributions by series and regime label.

    Parameters
    ----------
    durations:
        Duration table containing a ``duration`` column and a regime label.
    group_col:
        Series identifier column.
    regime_col:
        Regime label column supplied by the caller.

    Returns
    -------
    pd.DataFrame
        Summary statistics by series and regime.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required = {group_col, regime_col, "duration"}
    missing = required.difference(durations.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    summary = (
        durations.groupby([group_col, regime_col], dropna=False)["duration"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    return summary.rename(columns={"count": "n_durations"})
