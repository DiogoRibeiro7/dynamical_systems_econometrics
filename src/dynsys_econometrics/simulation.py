"""Simulation utilities for controlled dynamical and econometric examples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _validate_positive_int(value: int, name: str) -> None:
    """Validate that a value is a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_non_negative_int(value: int, name: str) -> None:
    """Validate that a value is a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


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
    _validate_positive_int(n_steps, "n_steps")
    _validate_non_negative_int(burn_in, "burn_in")
    if not 0.0 < x0 < 1.0:
        raise ValueError("x0 must be in the open interval (0, 1).")
    if not 0.0 < r <= 4.0:
        raise ValueError("r must be in the interval (0, 4].")

    total_steps = n_steps + burn_in
    values = np.empty(total_steps, dtype=np.float64)
    values[0] = x0

    for idx in range(1, total_steps):
        values[idx] = r * values[idx - 1] * (1.0 - values[idx - 1])

    return values[burn_in:]


def simulate_iid_gaussian(
    n_steps: int,
    mean: float = 0.0,
    std: float = 1.0,
    seed: int | None = 42,
) -> FloatArray:
    """Simulate iid Gaussian draws.

    Parameters
    ----------
    n_steps:
        Number of observations.
    mean:
        Distribution mean.
    std:
        Distribution standard deviation (strictly positive).
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_steps,)``.
    """
    _validate_positive_int(n_steps, "n_steps")
    if std <= 0.0:
        raise ValueError("std must be positive.")

    rng = np.random.default_rng(seed)
    return rng.normal(loc=mean, scale=std, size=n_steps).astype(np.float64)


def simulate_student_t(
    n_steps: int,
    df: int = 5,
    mean: float = 0.0,
    scale: float = 1.0,
    seed: int | None = 42,
) -> FloatArray:
    """Simulate iid Student-t draws.

    Parameters
    ----------
    n_steps:
        Number of observations.
    df:
        Degrees of freedom for Student-t (must be positive).
    mean:
        Location shift.
    scale:
        Scale parameter (strictly positive).
    seed:
        Optional RNG seed for reproducibility.
    """
    _validate_positive_int(n_steps, "n_steps")
    if not isinstance(df, int) or df <= 1:
        raise ValueError("df must be an integer greater than 1.")
    if scale <= 0.0:
        raise ValueError("scale must be positive.")

    rng = np.random.default_rng(seed)
    return (mean + scale * rng.standard_t(df=df, size=n_steps)).astype(np.float64)


def simulate_ar1(
    n_steps: int,
    phi: float = 0.8,
    sigma: float = 1.0,
    mean: float = 0.0,
    seed: int | None = 42,
) -> FloatArray:
    """Simulate a Gaussian AR(1) process for econometric comparison.

    This is a baseline weakly dependent process. It lets us compare tail clustering from
    standard econometric dependence against tail clustering induced by nonlinear dynamics.
    """
    _validate_positive_int(n_steps, "n_steps")
    if abs(phi) >= 1.0:
        raise ValueError("phi must satisfy abs(phi) < 1 for stationarity.")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")

    rng = np.random.default_rng(seed)
    shocks = rng.normal(loc=0.0, scale=sigma, size=n_steps)
    values = np.empty(n_steps, dtype=np.float64)
    values[0] = mean + shocks[0]

    for idx in range(1, n_steps):
        values[idx] = mean + phi * values[idx - 1] + shocks[idx]

    return values


def simulate_garch11(
    n_steps: int,
    omega: float = 0.01,
    alpha: float = 0.05,
    beta: float = 0.9,
    mean: float = 0.0,
    seed: int | None = 42,
) -> FloatArray:
    """Simulate a Gaussian GARCH(1,1) sequence.

    Parameters
    ----------
    n_steps:
        Number of observations.
    omega:
        Innovation variance intercept (positive).
    alpha:
        ARCH coefficient for lagged squared innovation (non-negative).
    beta:
        GARCH coefficient for lagged variance (non-negative).
    mean:
        Process location.
    seed:
        Optional RNG seed for reproducibility.
    """
    _validate_positive_int(n_steps, "n_steps")
    if omega <= 0.0:
        raise ValueError("omega must be positive.")
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")
    if alpha + beta >= 1.0:
        raise ValueError("alpha + beta must be strictly less than 1.")

    rng = np.random.default_rng(seed)
    innovations = np.empty(n_steps, dtype=np.float64)
    conditional_var = np.empty(n_steps, dtype=np.float64)
    eps = np.empty(n_steps, dtype=np.float64)

    unconditional_var = omega / (1.0 - alpha - beta)
    conditional_var[0] = unconditional_var
    eps[0] = rng.normal(scale=np.sqrt(conditional_var[0]))
    innovations[0] = mean + eps[0]

    for step in range(1, n_steps):
        conditional_var[step] = omega + alpha * (eps[step - 1] ** 2.0) + beta * conditional_var[step - 1]
        if conditional_var[step] <= 0.0:
            raise RuntimeError("Non-positive conditional variance encountered.")
        eps[step] = rng.normal(scale=np.sqrt(conditional_var[step]))
        innovations[step] = mean + eps[step]

    return innovations


def simulate_pomeau_manneville(
    n_steps: int,
    x0: float = 1e-3,
    a: float = 1.0,
    z: float = 2.0,
    burn_in: int = 100,
) -> FloatArray:
    """Simulate the Pomeau-Manneville intermittent map.

    The map is
    ``x_{t+1} = (x_t + a * x_t**z) mod 1`` with ``x_t`` in ``[0, 1)``.
    """
    _validate_positive_int(n_steps, "n_steps")
    _validate_non_negative_int(burn_in, "burn_in")
    if not 0.0 < x0 < 1.0:
        raise ValueError("x0 must be in the open interval (0, 1).")
    if a <= 0.0:
        raise ValueError("a must be positive.")
    if z <= 1.0:
        raise ValueError("z must be greater than 1.")

    total_steps = n_steps + burn_in
    values = np.empty(total_steps, dtype=np.float64)
    values[0] = x0

    for idx in range(1, total_steps):
        values[idx] = (values[idx - 1] + a * (values[idx - 1] ** z)) % 1.0

    return values[burn_in:]


def simulate_coupled_logistic_maps(
    n_steps: int,
    n_series: int = 2,
    coupling: float = 0.05,
    r: float | NDArray[np.float64] = 4.0,
    x0: NDArray[np.float64] | None = None,
    burn_in: int = 200,
    seed: int | None = 42,
) -> FloatArray:
    """Simulate coupled logistic maps.

    Dynamics:
    ``x_i(t+1) = (1-c) f_i(x_i(t)) + c * mean_j f_j(x_j(t))``
    with ``f_i(x)=r_i x (1-x)``.
    """
    _validate_positive_int(n_steps, "n_steps")
    if not isinstance(n_series, int) or n_series <= 0:
        raise ValueError("n_series must be a positive integer.")
    if not 0.0 <= coupling <= 1.0:
        raise ValueError("coupling must be in [0, 1].")
    _validate_non_negative_int(burn_in, "burn_in")

    if isinstance(r, float):
        if not 0.0 < r <= 4.0:
            raise ValueError("r must be in the interval (0, 4].")
        r_values = np.full(n_series, r, dtype=np.float64)
    else:
        r_values = np.asarray(r, dtype=np.float64)
        if r_values.shape != (n_series,):
            raise ValueError("r must be a float or an array with shape (n_series,).")
        if np.any(r_values <= 0.0) or np.any(r_values > 4.0):
            raise ValueError("all r values must be in (0, 4].")

    if x0 is None:
        rng = np.random.default_rng(seed)
        x = rng.uniform(low=1e-6, high=1.0 - 1e-6, size=n_series)
    else:
        x = np.asarray(x0, dtype=np.float64).copy()
        if x.shape != (n_series,):
            raise ValueError("x0 must be None or an array with shape (n_series,).")
        if np.any(x <= 0.0) or np.any(x >= 1.0):
            raise ValueError("all x0 values must be in the open interval (0, 1).")

    total_steps = n_steps + burn_in
    trajectories = np.empty((total_steps, n_series), dtype=np.float64)
    trajectories[0] = x

    for step in range(1, total_steps):
        map_output = r_values * trajectories[step - 1] * (1.0 - trajectories[step - 1])
        local = (1.0 - coupling) * map_output + coupling * map_output.mean()
        trajectories[step] = local

    return trajectories[burn_in:]
