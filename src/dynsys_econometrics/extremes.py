"""Extreme-value utilities for dependent econometric sequences."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


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


def exceedance_indicator(values: FloatArray, threshold: float) -> BoolArray:
    """Return a boolean indicator for exceedances above a threshold."""
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")
    if values.size == 0:
        raise ValueError("values must not be empty.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    return values > threshold


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
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")
    if not 0.0 < threshold_quantile < 1.0:
        raise ValueError("threshold_quantile must be between 0 and 1.")
    if not isinstance(run_length, int) or run_length < 1:
        raise ValueError("run_length must be a positive integer.")

    threshold = float(np.quantile(values, threshold_quantile))
    exceedances = exceedance_indicator(values, threshold)
    exceedance_positions = np.flatnonzero(exceedances)

    if exceedance_positions.size == 0:
        return ExtremalIndexResult(threshold, run_length, 0, 0, float("nan"))

    clusters = 1
    previous_position = int(exceedance_positions[0])

    for position in exceedance_positions[1:]:
        gap = int(position) - previous_position
        if gap > run_length:
            clusters += 1
        previous_position = int(position)

    theta_hat = clusters / exceedance_positions.size
    return ExtremalIndexResult(
        threshold=threshold,
        run_length=run_length,
        n_exceedances=int(exceedance_positions.size),
        n_clusters=int(clusters),
        theta_hat=float(theta_hat),
    )
