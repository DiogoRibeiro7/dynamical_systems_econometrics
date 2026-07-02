"""Tools for dynamical-systems inspired econometric rare-event analysis."""

from dynsys_econometrics.extremes import ExtremalIndexResult, estimate_runs_extremal_index
from dynsys_econometrics.return_times import compute_return_times
from dynsys_econometrics.simulation import logistic_map, simulate_ar1

__all__ = [
    "ExtremalIndexResult",
    "compute_return_times",
    "estimate_runs_extremal_index",
    "logistic_map",
    "simulate_ar1",
]
