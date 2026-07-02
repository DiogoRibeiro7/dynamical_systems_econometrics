"""Extreme-value utilities for dependent econometric sequences."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


@dataclass(frozen=True)
class ExtremalIndexResult:
    """Result from a runs estimator of the extremal index.

    The extremal index is near 1 when extremes behave almost independently. It is lower
    than 1 when extremes tend to cluster, which is central in dynamical-systems extreme
    value theory and highly relevant for crisis econometrics.
    """

    threshold: float
    run_length: int
    n_exceedances: int
    n_clusters: int
    theta_hat: float


@dataclass(frozen=True)
class ExceedanceExtractionResult:
    """Container with exceedance extraction outputs."""

    threshold: float
    indicator: BoolArray
    indices: IntArray
    values: FloatArray


@dataclass(frozen=True)
class PeaksOverThresholdResult:
    """Container with exceedance excesses above a threshold."""

    threshold: float
    indices: IntArray
    exceedance_values: FloatArray
    excesses: FloatArray


@dataclass(frozen=True)
class ThresholdSensitivityResult:
    """Container for extremal-index sensitivity across thresholds."""

    quantile: float
    threshold: float
    n_exceedances: int
    n_clusters: int
    theta_runs: float
    theta_cluster_mean: float


def _coerce_numeric_series(values: Sequence[float] | FloatArray) -> FloatArray:
    """Validate and convert input to a finite 1-D float64 NumPy array."""
    if pd is not None and isinstance(values, pd.Series):
        arr = values.to_numpy(dtype=np.float64)
    elif isinstance(values, np.ndarray):
        arr = np.asarray(values, dtype=np.float64)
    else:
        raise TypeError("values must be a one-dimensional np.ndarray or pd.Series.")

    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if arr.size == 0:
        raise ValueError("values must not be empty.")
    if not np.isfinite(arr).all():
        raise ValueError("values must contain only finite numbers.")

    return arr.astype(np.float64, copy=False)


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_non_negative_int(value: int, name: str) -> None:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _run_length_validation(run_length: int) -> None:
    """Validate declustering run length."""
    if not isinstance(run_length, int) or run_length < 1:
        raise ValueError("run_length must be a positive integer.")


def exceedance_indicator(values: FloatArray, threshold: float) -> BoolArray:
    """Return a boolean indicator for exceedances above a threshold."""
    values = _coerce_numeric_series(values)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    return values > threshold


def extract_threshold_exceedances(
    values: FloatArray,
    threshold: float,
) -> ExceedanceExtractionResult:
    """Extract exceedance metadata for a fixed threshold.

    Parameters
    ----------
    values:
        One-dimensional series.
    threshold:
        Numeric threshold used to define exceedances.
    """
    indicator = exceedance_indicator(values, threshold)
    values_arr = _coerce_numeric_series(values)
    indices = np.flatnonzero(indicator)
    exceedance_values = values_arr[indicator]
    return ExceedanceExtractionResult(
        threshold=float(threshold),
        indicator=indicator,
        indices=indices.astype(np.int64, copy=False),
        values=exceedance_values,
    )


def block_maxima(
    values: FloatArray,
    block_size: int = 50,
    drop_incomplete_block: bool = True,
) -> FloatArray:
    """Compute block maxima over disjoint blocks.

    Parameters
    ----------
    values:
        One-dimensional numeric array.
    block_size:
        Number of points per block.
    drop_incomplete_block:
        If True, drop a trailing incomplete block.
    """
    values_arr = _coerce_numeric_series(values)
    _validate_positive_int(block_size, "block_size")

    if drop_incomplete_block:
        n_complete = values_arr.size // block_size
        if n_complete == 0:
            raise ValueError("not enough observations for a complete block.")
        usable = values_arr[: n_complete * block_size]
        reshaped = usable.reshape(n_complete, block_size)
        return np.max(reshaped, axis=1)

    output = []
    start = 0
    while start < values_arr.size:
        output.append(float(np.max(values_arr[start : start + block_size])))
        start += block_size
    return np.asarray(output, dtype=np.float64)


def peaks_over_threshold(
    values: FloatArray,
    threshold: float,
) -> PeaksOverThresholdResult:
    """Compute peaks-over-threshold values and excesses."""
    result = extract_threshold_exceedances(values, threshold)
    excesses = result.values - result.threshold
    return PeaksOverThresholdResult(
        threshold=result.threshold,
        indices=result.indices,
        exceedance_values=result.values,
        excesses=excesses,
    )


def runs_declustering_from_indicator(
    exceedance_indicator: BoolArray,
    run_length: int = 5,
) -> IntArray:
    """Convert a boolean exceedance array into cluster sizes by runs rule."""
    if exceedance_indicator.ndim != 1:
        raise ValueError("exceedance_indicator must be one-dimensional.")
    if exceedance_indicator.size == 0:
        raise ValueError("exceedance_indicator must not be empty.")
    if not np.issubdtype(exceedance_indicator.dtype, np.bool_):
        raise ValueError("exceedance_indicator must be boolean.")
    _run_length_validation(run_length)

    positions = np.flatnonzero(exceedance_indicator)
    if positions.size == 0:
        return np.zeros(0, dtype=np.int64)

    cluster_size = 1
    previous = int(positions[0])
    clusters: list[int] = []

    for position in positions[1:]:
        gap = int(position) - previous
        if gap > run_length:
            clusters.append(cluster_size)
            cluster_size = 1
        else:
            cluster_size += 1
        previous = int(position)

    clusters.append(cluster_size)
    return np.asarray(clusters, dtype=np.int64)


def runs_declustering(
    values: FloatArray,
    threshold: float,
    run_length: int = 5,
) -> IntArray:
    """Estimate run-based cluster sizes for exceedances."""
    _run_length_validation(run_length)
    indicator = exceedance_indicator(values, threshold)
    return runs_declustering_from_indicator(indicator, run_length=run_length)


def estimate_cluster_size_extremal_index(
    values: FloatArray,
    threshold: float,
    run_length: int = 5,
) -> float:
    """Estimate the extremal index as inverse mean cluster size."""
    _run_length_validation(run_length)
    cluster_sizes = runs_declustering(values, threshold, run_length=run_length)
    if cluster_sizes.size == 0:
        return float("nan")

    return 1.0 / float(np.mean(cluster_sizes))


def estimate_runs_extremal_index(
    values: FloatArray,
    threshold_quantile: float = 0.95,
    run_length: int = 5,
) -> ExtremalIndexResult:
    """Estimate the extremal index using a simple runs declustering estimator.

    Parameters
    ----------
    values:
        One-dimensional time series.
    threshold_quantile:
        Quantile used to define rare events.
    run_length:
        Number of consecutive non-exceedances needed to close a cluster.

    Returns
    -------
    ExtremalIndexResult
        Contains threshold, exceedance count, cluster count, and theta estimate.
    """
    _run_length_validation(run_length)
    values_arr = _coerce_numeric_series(values)
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    threshold = float(np.quantile(values_arr, threshold_quantile))
    cluster_sizes = runs_declustering(values_arr, threshold=threshold, run_length=run_length)
    n_exceedances = int(np.flatnonzero(values_arr > threshold).size)
    if n_exceedances == 0:
        return ExtremalIndexResult(threshold, run_length, 0, 0, float("nan"))

    n_clusters = int(cluster_sizes.size)
    theta_hat = float(n_clusters / n_exceedances)
    return ExtremalIndexResult(
        threshold=threshold,
        run_length=run_length,
        n_exceedances=n_exceedances,
        n_clusters=n_clusters,
        theta_hat=float(theta_hat),
    )


def estimate_extremal_index(
    values: FloatArray,
    threshold_quantile: float = 0.95,
    run_length: int = 5,
    method: str = "runs",
) -> float:
    """Estimate the extremal index with a simple method selector.

    Methods:
      - ``"runs"``: number of clusters divided by exceedance count.
      - ``"cluster_size_mean"``: inverse of mean cluster size.
    """
    if method not in {"runs", "cluster_size_mean"}:
        raise ValueError("method must be 'runs' or 'cluster_size_mean'.")
    if method == "runs":
        return estimate_runs_extremal_index(
            values,
            threshold_quantile=threshold_quantile,
            run_length=run_length,
        ).theta_hat

    values_arr = _coerce_numeric_series(values)
    _validate_positive_int(run_length, "run_length")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    threshold = float(np.quantile(values_arr, threshold_quantile))
    return estimate_cluster_size_extremal_index(values_arr, threshold, run_length=run_length)


def threshold_sensitivity_analysis(
    values: FloatArray,
    threshold_quantiles: Sequence[float] = (0.8, 0.9, 0.925, 0.95, 0.975, 0.99),
    run_length: int = 5,
) -> tuple[ThresholdSensitivityResult, ...]:
    """Evaluate extremal-index estimates across a grid of quantiles.

    Returns a tuple of results sorted by threshold quantile.
    """
    values_arr = _coerce_numeric_series(values)
    _run_length_validation(run_length)

    if not isinstance(threshold_quantiles, Sequence) or len(threshold_quantiles) == 0:
        raise ValueError("threshold_quantiles must be a non-empty sequence.")

    quantiles = [float(q) for q in threshold_quantiles]
    if any((q <= 0.0) or (q >= 1.0) for q in quantiles):
        raise ValueError("all threshold_quantiles must be in (0, 1).")

    results = []
    for q in quantiles:
        runs_result = estimate_runs_extremal_index(
            values_arr,
            threshold_quantile=q,
            run_length=run_length,
        )
        if runs_result.n_exceedances == 0:
            theta_cluster = float("nan")
        else:
            threshold = runs_result.threshold
            theta_cluster = estimate_cluster_size_extremal_index(
                values_arr,
                threshold=threshold,
                run_length=run_length,
            )

        results.append(
            ThresholdSensitivityResult(
                quantile=q,
                threshold=runs_result.threshold,
                n_exceedances=runs_result.n_exceedances,
                n_clusters=runs_result.n_clusters,
                theta_runs=runs_result.theta_hat,
                theta_cluster_mean=theta_cluster,
            )
        )

    return tuple(results)
