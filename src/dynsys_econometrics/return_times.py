"""Hitting-time and return-time utilities for rare-event regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
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
