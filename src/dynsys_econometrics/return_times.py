"""Hitting-time and return-time utilities for rare-event regions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def compute_return_times(values: FloatArray, threshold: float) -> IntArray:
    """Compute gaps between consecutive threshold exceedances.

    In econometric applications, these gaps represent the recurrence time between crisis
    episodes, stress periods, volatility bursts, or other rare-event regions.
    """
    if values.ndim != 1:
        raise ValueError("values must be a one-dimensional array.")
    if values.size == 0:
        raise ValueError("values must not be empty.")
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")

    positions = np.flatnonzero(values > threshold)
    if positions.size < 2:
        return np.array([], dtype=np.int64)

    return np.diff(positions).astype(np.int64)
