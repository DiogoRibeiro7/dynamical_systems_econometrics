"""Simulation utilities for controlled dynamical and econometric examples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def logistic_map(
    n_steps: int,
    x0: float = 0.123,
    r: float = 4.0,
    burn_in: int = 100,
) -> FloatArray:
    """Simulate the logistic map x[t+1] = r * x[t] * (1 - x[t]).

    Parameters
    ----------
    n_steps:
        Number of retained observations.
    x0:
        Initial state in the open interval (0, 1).
    r:
        Logistic-map parameter. The canonical chaotic case is r=4.
    burn_in:
        Number of initial iterations discarded to reduce dependence on x0.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_steps,)``.
    """
    if not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if not 0.0 < x0 < 1.0:
        raise ValueError("x0 must be in the open interval (0, 1).")
    if not 0.0 < r <= 4.0:
        raise ValueError("r must be in the interval (0, 4].")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative.")

    total_steps = n_steps + burn_in
    values = np.empty(total_steps, dtype=np.float64)
    values[0] = x0

    for idx in range(1, total_steps):
        values[idx] = r * values[idx - 1] * (1.0 - values[idx - 1])

    return values[burn_in:]


def simulate_ar1(
    n_steps: int,
    phi: float = 0.8,
    sigma: float = 1.0,
    seed: int | None = 42,
) -> FloatArray:
    """Simulate a Gaussian AR(1) process for econometric comparison.

    This is a baseline weakly dependent process. It lets us compare tail clustering from
    standard econometric dependence against tail clustering induced by nonlinear dynamics.
    """
    if not isinstance(n_steps, int) or n_steps <= 0:
        raise ValueError("n_steps must be a positive integer.")
    if abs(phi) >= 1.0:
        raise ValueError("phi must satisfy abs(phi) < 1 for stationarity.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=0.0, scale=sigma, size=n_steps)
    values = np.empty(n_steps, dtype=np.float64)
    values[0] = shocks[0]

    for idx in range(1, n_steps):
        values[idx] = phi * values[idx - 1] + shocks[idx]

    return values
