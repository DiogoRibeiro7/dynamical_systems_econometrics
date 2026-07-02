"""Tools for dynamical-systems inspired econometric rare-event analysis."""

from dynsys_econometrics.extremes import ExtremalIndexResult, estimate_runs_extremal_index
from dynsys_econometrics.return_times import compute_return_times
from dynsys_econometrics.simulation import (
    logistic_map,
    simulate_ar1,
    simulate_coupled_logistic_maps,
    simulate_garch11,
    simulate_iid_gaussian,
    simulate_pomeau_manneville,
    simulate_student_t,
)

__all__ = [
    "ExtremalIndexResult",
    "compute_return_times",
    "estimate_runs_extremal_index",
    "logistic_map",
    "simulate_ar1",
    "simulate_coupled_logistic_maps",
    "simulate_garch11",
    "simulate_iid_gaussian",
    "simulate_pomeau_manneville",
    "simulate_student_t",
]
